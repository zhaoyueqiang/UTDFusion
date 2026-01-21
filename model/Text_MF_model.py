import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
from einops import rearrange
import clip


class PromptCrossAttentionAggregator(nn.Module):
    """
    用于在多退化提示 (F_list) 与统一提示 (F_unified) 之间做 Cross-Attention 聚合，
    得到教师表示 F_teacher。

    参数:
        d_model  : 文本编码器输出维度（如 CLIP-ViT-B/32 为 512）
        n_heads  : Multi-Head Attention 头数
        d_ff     : 前馈网络隐藏层维度
        dropout  : dropout 概率
    输入:
        F_unified: [B, d_model]，统一提示的文本向量
        F_list   : [B, N, d_model]，N 个子任务提示的文本向量
    输出:
        F_teacher: [B, d_model]，聚合后的教师表示
        attn_weights: [B, 1, N]，注意力权重（可用于可视化分析）
    """

    def __init__(self, d_model, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # 多头注意力模块
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model,
                                                num_heads=n_heads,
                                                dropout=dropout,
                                                batch_first=False)  # 输入: [seq_len, batch, dim]

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 前馈网络（FFN）
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # 可选输出映射层（保证输出维度与输入一致）
        self.proj_out = nn.Linear(d_model, d_model)

    def forward(self, F_unified, F_list):
        """
        Args:
            F_unified: [B, d_model]
            F_list:    [B, N, d_model]
        Returns:
            F_teacher: [B, d_model]
            attn_weights: [B, 1, N]
        """
        B, N, d = F_list.shape
        assert d == self.d_model, f"Expected {self.d_model}, got {d}"
        assert F_unified.shape == (B, d)

        # 调整形状以适配 nn.MultiheadAttention:
        # Q: [1, B, d] （seq_len=1，因为只有一个统一提示）
        # K, V: [N, B, d] （seq_len=N，因为有 N 个子任务提示）
        q = F_unified.unsqueeze(1).transpose(0, 1)  # [1, B, d]
        kv = F_list.transpose(0, 1)                 # [N, B, d]

        # Cross-Attention 计算
        attn_out, attn_weights = self.cross_attn(q, kv, kv)  # attn_out: [1, B, d]
        attn_out = attn_out.squeeze(0)                       # [B, d]
        attn_weights = attn_weights.permute(1, 0, 2)         # [B, 1, N]

        # 残差连接 + 层归一化
        x = self.norm1(F_unified + attn_out)

        # 前馈网络 + 残差连接 + 层归一化
        x2 = self.ffn(x)
        x = self.norm2(x + x2)

        # 可选线性映射
        F_teacher = self.proj_out(x)  # [B, d_model]

        return F_teacher, attn_weights

class TextGuidedAttention(nn.Module):
    def __init__(self, d_vis, d_text, n_heads=8, dropout=0.1):
        """
        d_vis  : 视觉特征维度
        d_text : 文本特征维度 (如 CLIP 的输出)
        n_heads: 注意力头数
        """
        super().__init__()
        self.n_heads = n_heads
        self.d_vis = d_vis
        self.head_dim = d_vis // n_heads
        assert d_vis % n_heads == 0, "d_vis must be divisible by n_heads"
        self.scale = self.head_dim ** -0.5

        # 文本投影到视觉空间
        self.text_proj = nn.Sequential(
            nn.Linear(d_text, d_vis),
            nn.GELU(),
            nn.Linear(d_vis, d_vis)  # 这样就能直接和视觉 KV 匹配
        )

        # Q/K/V projection
        self.q_proj = nn.Linear(d_vis, d_vis)
        self.k_proj = nn.Linear(d_vis, d_vis)
        self.v_proj = nn.Linear(d_vis, d_vis)

        self.out_proj = nn.Linear(d_vis, d_vis)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_vis)

        # 可选 gating
        self.gate = StudentTeacherJointGate(d_text,d_vis)

    def _split_heads(self, x):
        # [B, T, D] -> [B, H, T, Hd]
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

    def _combine_heads(self, x):
        # [B, H, T, Hd] -> [B, T, D]
        B, H, T, Hd = x.shape
        return x.permute(0, 2, 1, 3).reshape(B, T, H * Hd)

    def forward(self, F_vis, F_text,F_unified, F_teacher):
        """
        F_vis: [B, T, D_vis]   - 融合视觉特征
        F_text: [B, D_text]    - 文本统一特征 F_unified
        """
        
        # 四维特征图 [B, C, H, W]
        B, C, H, W = F_vis.shape
        F_vis = F_vis.view(B, C, H*W).permute(0, 2, 1).contiguous()  # [B, H*W, C]
        B, T, Dv = F_vis.shape

        # 文本投影成视觉空间 Q token
        F_text_proj = self.text_proj(F_text).unsqueeze(1)  # [B, 1, D_vis]

        Q = self.q_proj(F_text_proj)      # [B, 1, D_vis]
        K = self.k_proj(F_vis)            # [B, T, D_vis]
        V = self.v_proj(F_vis)            # [B, T, D_vis]

        Qh = self._split_heads(Q)         # [B, H, 1, Hd]
        Kh = self._split_heads(K)         # [B, H, T, Hd]
        Vh = self._split_heads(V)         # [B, H, T, Hd]

        attn_scores = torch.matmul(Qh, Kh.transpose(-2, -1)) * self.scale  # [B, H, 1, T]
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        Zh = torch.matmul(attn_weights, Vh)  # [B, H, 1, Hd]
        Z = self._combine_heads(Zh)          # [B, 1, D_vis]
        Z = Z.expand(B, T, Dv)               # 扩展到视觉 token 数

        # 残差 + Norm + Gate
        Z = self.out_proj(Z)
        g = self.gate(F_unified, F_teacher).unsqueeze(1).expand(B, T, Dv)  # gate 来自文本
        out = self.norm(F_vis + g * Z)

        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)  # [B, C, H, W]

        attn_map = attn_weights.squeeze(2)  # [B, n_heads, T]
        attn_map = attn_map.view(B, self.n_heads, H, W)  # [B, n_heads, H, W]

        return out, attn_weights
    
class StudentTeacherJointGate(nn.Module):
    def __init__(self, d_text, d_vis):
        super().__init__()
        self.fuse_attn = nn.Linear(d_text, 1)
        self.gate_proj = nn.Linear(d_text, d_vis)  # 投影到视觉特征维度
        self.sigmoid = nn.Sigmoid()

    def forward(self, F_unified, F_teacher=None):
        if F_teacher is None:
            F_cond = F_unified
        else:
            F_stack = torch.stack([F_unified, F_teacher], dim=1)  # [B, 2, d_text]
            attn_score = torch.softmax(self.fuse_attn(F_stack), dim=1)
            F_cond = torch.sum(attn_score * F_stack, dim=1)

        g = self.sigmoid(self.gate_proj(F_cond))  # [B, d_vis]
        return g

class Text_MF(nn.Module):
    def __init__(self, model_clip, inp_A_channels=3, inp_B_channels=3, out_channels=3,
                 dim=48, num_blocks=[2, 2, 2, 2],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Text_MF, self).__init__()

        self.model_clip = model_clip
        self.model_clip.eval()

        self.PromptCrossAtten=PromptCrossAttentionAggregator(d_model=512, n_heads=8, d_ff=2048, dropout=0.1)

        self.encoder_A = Encoder_A(inp_channels=inp_A_channels, dim=dim, num_blocks=num_blocks, heads=heads,
                                   ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        self.encoder_B = Encoder_B(inp_channels=inp_B_channels, dim=dim, num_blocks=num_blocks, heads=heads,
                                   ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        self.cross_attention = Cross_attention(dim * 2 ** 3)
        self.attention_spatial = Attention_spatial(dim * 2 ** 3)

        self.feature_fusion_4 = Fusion_Embed(embed_dim=dim * 2 ** 3)
        # self.prompt_guidance_4 = FeatureWiseAffine(in_channels=512, out_channels=dim * 2 ** 3)
        self.prompt_guidance_4 = TextGuidedAttention(dim * 2 ** 3,512)
        self.decoder_level4 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.feature_fusion_3 = Fusion_Embed(embed_dim = dim * 2 ** 2)
        # self.prompt_guidance_3 = FeatureWiseAffine(in_channels=512, out_channels=dim * 2 ** 2)
        self.prompt_guidance_3 = TextGuidedAttention(dim * 2 ** 2,512)
        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.feature_fusion_2 = Fusion_Embed(embed_dim = dim * 2 ** 1)
        # self.prompt_guidance_2 = FeatureWiseAffine(in_channels=512, out_channels=dim * 2 ** 1)
        self.prompt_guidance_2 = TextGuidedAttention(dim * 2 ** 1,512)
        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.feature_fusion_1 = Fusion_Embed(embed_dim = dim)
        # self.prompt_guidance_1 = FeatureWiseAffine(in_channels=512, out_channels=dim)
        self.prompt_guidance_1 = TextGuidedAttention(dim,512)
        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img_A, inp_img_B, text,type="teacher",device='cuda'):
        b = inp_img_A.shape[0]
        F_teacher = None
        F_unified = None
        text_features = self.get_text_feature(text.expand(b, -1)).to(inp_img_A.dtype)
        F_unified = text_features
        if type=="student":
            T_list = ["This is the infrared visible light fusion task. Visible images have the low light degradation.", 
                      "This is the infrared visible light fusion task. Infrared images have the noise degradation.", 
                      "This is the infrared visible light fusion task. Infrared images have the low contrast degradation."]
            # text = clip.tokenize(text_line).to(device)
            # for t in T_list:
            #     t = clip.tokenize(t).to(device)
            F_list = torch.stack([self.get_text_feature(clip.tokenize(t).to(device).expand(b, -1)).to(inp_img_A.dtype) for t in T_list], dim=1)
            with torch.no_grad():
                F_teacher, attn_w = self.PromptCrossAtten(F_unified.detach(), F_list.detach())
            # F_teacher = torch.stack(F_list).mean(dim=0)
        

        out_enc_level4_A, out_enc_level3_A, out_enc_level2_A, out_enc_level1_A = self.encoder_A(inp_img_A)
        out_enc_level4_B, out_enc_level3_B, out_enc_level2_B, out_enc_level1_B = self.encoder_B(inp_img_B)

        out_enc_level4_A, out_enc_level4_B = self.cross_attention(out_enc_level4_A, out_enc_level4_B)
        out_enc_level4 = self.feature_fusion_4(out_enc_level4_A, out_enc_level4_B)
        out_enc_level4 = self.attention_spatial(out_enc_level4)
        # print(out_enc_level4.shape)
        # print(text_features.shape)

        out_enc_level4,_ = self.prompt_guidance_4(out_enc_level4, text_features, F_unified, F_teacher)
        inp_dec_level4 = out_enc_level4

        out_dec_level4 = self.decoder_level4(inp_dec_level4)

        inp_dec_level3 = self.up4_3(out_dec_level4)
        inp_dec_level3,_ = self.prompt_guidance_3(inp_dec_level3, text_features, F_unified, F_teacher)
        out_enc_level3 = self.feature_fusion_3(out_enc_level3_A, out_enc_level3_B)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2,_ = self.prompt_guidance_2(inp_dec_level2, text_features, F_unified, F_teacher)
        out_enc_level2 = self.feature_fusion_2(out_enc_level2_A, out_enc_level2_B)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1,_ = self.prompt_guidance_1(inp_dec_level1, text_features, F_unified, F_teacher)
        out_enc_level1 = self.feature_fusion_1(out_enc_level1_A, out_enc_level1_B)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1, F_unified, F_teacher

    @torch.no_grad()
    def get_text_feature(self, text):
        text_feature = self.model_clip.encode_text(text)
        return text_feature


class Cross_attention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()
        self.n_head = n_head
        self.norm_A = nn.GroupNorm(norm_groups, in_channel)
        self.norm_B = nn.GroupNorm(norm_groups, in_channel)
        self.qkv_A = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_A = nn.Conv2d(in_channel, in_channel, 1)

        self.qkv_B = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_B = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, x_A, x_B):
        batch, channel, height, width = x_A.shape

        n_head = self.n_head
        head_dim = channel // n_head

        x_A = self.norm_A(x_A)
        qkv_A = self.qkv_A(x_A).view(batch, n_head, head_dim * 3, height, width)
        query_A, key_A, value_A = qkv_A.chunk(3, dim=2)

        x_B = self.norm_B(x_B)
        qkv_B = self.qkv_B(x_B).view(batch, n_head, head_dim * 3, height, width)
        query_B, key_B, value_B = qkv_B.chunk(3, dim=2)

        attn_A = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_B, key_A
        ).contiguous() / math.sqrt(channel)
        attn_A = attn_A.view(batch, n_head, height, width, -1)
        attn_A = torch.softmax(attn_A, -1)
        attn_A = attn_A.view(batch, n_head, height, width, height, width)

        out_A = torch.einsum("bnhwyx, bncyx -> bnchw", attn_A, value_A).contiguous()
        out_A = self.out_A(out_A.view(batch, channel, height, width))
        out_A = out_A + x_A

        attn_B = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_A, key_B
        ).contiguous() / math.sqrt(channel)
        attn_B = attn_B.view(batch, n_head, height, width, -1)
        attn_B = torch.softmax(attn_B, -1)
        attn_B = attn_B.view(batch, n_head, height, width, height, width)

        out_B = torch.einsum("bnhwyx, bncyx -> bnchw", attn_B, value_B).contiguous()
        out_B = self.out_B(out_B.view(batch, channel, height, width))
        out_B = out_B + x_B

        return out_A, out_B

class Attention_spatial(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head
        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)
        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input

##########################################################################
## Feature Modulation
class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, text_embed):
        text_embed = text_embed.unsqueeze(1)
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.MLP(text_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        return x


class Encoder_A(nn.Module):
    def __init__(self, inp_channels=3, dim=32, num_blocks=[2, 3, 3, 4], heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=False,
                 LayerNorm_type='WithBias'):
        super(Encoder_A, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.encoder_level4 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

    def forward(self, inp_img_A):
        inp_enc_level1_A = self.patch_embed(inp_img_A)
        out_enc_level1_A = self.encoder_level1(inp_enc_level1_A)

        inp_enc_level2_A = self.down1_2(out_enc_level1_A)
        out_enc_level2_A = self.encoder_level2(inp_enc_level2_A)

        inp_enc_level3_A = self.down2_3(out_enc_level2_A)
        out_enc_level3_A = self.encoder_level3(inp_enc_level3_A)

        inp_enc_level4_A = self.down3_4(out_enc_level3_A)
        out_enc_level4_A = self.encoder_level4(inp_enc_level4_A)

        return out_enc_level4_A, out_enc_level3_A, out_enc_level2_A, out_enc_level1_A


class Encoder_B(nn.Module):
    def __init__(self, inp_channels=1, dim=32, num_blocks=[2, 3, 3, 4], heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(Encoder_B, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.encoder_level4 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

    def forward(self, inp_img_B):
        inp_enc_level1_B = self.patch_embed(inp_img_B)
        out_enc_level1_B = self.encoder_level1(inp_enc_level1_B)

        inp_enc_level2_B = self.down1_2(out_enc_level1_B)
        out_enc_level2_B = self.encoder_level2(inp_enc_level2_B)

        inp_enc_level3_B = self.down2_3(out_enc_level2_B)
        out_enc_level3_B = self.encoder_level3(inp_enc_level3_B)

        inp_enc_level4_B = self.down3_4(out_enc_level3_B)
        out_enc_level4_B = self.encoder_level4(inp_enc_level4_B)

        return out_enc_level4_B, out_enc_level3_B, out_enc_level2_B, out_enc_level1_B

class Fusion_Embed(nn.Module):
    def __init__(self, embed_dim, bias=False):
        super(Fusion_Embed, self).__init__()

        self.fusion_proj = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x_A, x_B):
        x = torch.concat([x_A, x_B], dim=1)
        x = self.fusion_proj(x)
        return x

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)