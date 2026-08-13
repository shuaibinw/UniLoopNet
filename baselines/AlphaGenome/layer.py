
#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


def conv_block(x, num_channels, width=5):
    """AlphaGenome风格的卷积块"""
    x = F.layer_norm(x, x.shape[-1:])  # RMSBatchNorm的简化版本
    x = F.gelu(x)
    if width == 1:
        x = nn.Linear(x.shape[-1], num_channels)(x)
    else:
        # 1D卷积
        x = x.transpose(-1, -2)  # 调整维度用于Conv1d
        x = F.conv1d(x, weight=torch.randn(num_channels, x.shape[1], width).to(x.device), padding=width//2)
        x = x.transpose(-1, -2)  # 调整回来
    return x


class DNAEmbedder(nn.Module):
    """DNA序列嵌入器"""
    def __init__(self, input_dim=4, embed_dim=768):
        super().__init__()
        self.initial_conv = nn.Conv1d(input_dim, embed_dim, kernel_size=15, padding=7)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.gelu = nn.GELU()
        self.conv_layer = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2)
    
    def forward(self, x):
        out = self.initial_conv(x)
        out_transposed = out.transpose(1, 2)
        out_norm = self.layer_norm(out_transposed)
        out_norm = self.gelu(out_norm)
        out_norm = out_norm.transpose(1, 2)
        conv_out = self.conv_layer(out_norm)
        return out + conv_out


class DownResBlock(nn.Module):
    """下采样残差块"""
    def __init__(self, in_channels):
        super().__init__()
        self.out_channels = in_channels + 128
        self.layer_norm1 = nn.LayerNorm(in_channels)
        self.gelu1 = nn.GELU()
        self.conv1 = nn.Conv1d(in_channels, self.out_channels, kernel_size=5, padding=2)
        self.layer_norm2 = nn.LayerNorm(self.out_channels)
        self.gelu2 = nn.GELU()
        self.conv2 = nn.Conv1d(self.out_channels, self.out_channels, kernel_size=5, padding=2)
        
    def forward(self, x):
        x_transposed = x.transpose(1, 2)
        x_norm = self.layer_norm1(x_transposed)
        x_norm = self.gelu1(x_norm)
        x_norm = x_norm.transpose(1, 2)
        out = self.conv1(x_norm)
        x_padded = F.pad(x, (0, 0, 0, 128))
        out = out + x_padded
        out_transposed = out.transpose(1, 2)
        out_norm = self.layer_norm2(out_transposed)
        out_norm = self.gelu2(out_norm)
        out_norm = out_norm.transpose(1, 2)
        conv_out = self.conv2(out_norm)
        return out + conv_out


class SequenceEncoder(nn.Module):
    """AlphaGenome风格的序列编码器"""
    def __init__(self, seq_length=5000):
        super().__init__()
        self.dna_embedder = DNAEmbedder(input_dim=4, embed_dim=768)
        self.downres_blocks = nn.ModuleList()
        self.max_pools = nn.ModuleList()
        current_channels = 768
        for bin_size in [2, 4, 8, 16, 32, 64]:
            self.downres_blocks.append(DownResBlock(current_channels))
            current_channels += 128
            self.max_pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            
    def forward(self, x):
        intermediates = {}
        x = self.dna_embedder(x)
        intermediates['bin_size_1'] = x
        # print(f"bin_size_1: shape {x.shape}")
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        for i, (downres, pool) in enumerate(zip(self.downres_blocks, self.max_pools)):
            bin_size = 2 ** (i + 2)
            x = downres(x)
            intermediates[f'bin_size_{bin_size}'] = x
            # print(f"bin_size_{bin_size}: shape {x.shape}")
            if i < len(self.max_pools) - 1:  # 最后一层不池化
                x = pool(x)
        return x, intermediates


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, embed_dim=1536, num_heads=8, max_position=8192):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 1536 / 8 = 192
        self.max_position = max_position
        
        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, embed_dim)
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def apply_rope(self, x, positions=None):
        """旋转位置编码"""
        batch_size, seq_len, num_heads, head_dim = x.shape
        if positions is None:
            positions = torch.arange(seq_len, device=x.device)
        
        num_freq = head_dim // 2
        freqs = 1.0 / (10000 ** (torch.arange(num_freq, device=x.device) / num_freq))
        angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)
        angles = angles.repeat_interleave(2, dim=-1)
        
        cos_angles = torch.cos(angles).unsqueeze(0).unsqueeze(2)
        sin_angles = torch.sin(angles).unsqueeze(0).unsqueeze(2)
        
        cos_angles = cos_angles.expand(batch_size, -1, num_heads, -1)
        sin_angles = sin_angles.expand(batch_size, -1, num_heads, -1)

        x_pairs = x.view(batch_size, seq_len, num_heads, -1, 2)
        x_rotated = torch.stack([-x_pairs[..., 1], x_pairs[..., 0]], dim=-1)
        x_rotated = x_rotated.view(batch_size, seq_len, num_heads, head_dim)
        
        return x * cos_angles + x_rotated * sin_angles
        
    def forward(self, x, attention_bias=None):
        B, S, C = x.shape
        x_norm = self.layer_norm(x)
        
        q = self.q_proj(x_norm).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x_norm).view(B, S, self.num_heads, self.head_dim)
        v = self.v_proj(x_norm).view(B, S, self.num_heads, self.head_dim)
        
        q = self.apply_rope(q)
        k = self.apply_rope(k)
        
        scores = torch.einsum('bshc,bSkc->bhsS', q, k) / math.sqrt(self.head_dim)
        if attention_bias is not None:
            scores = scores + attention_bias
        scores = torch.tanh(scores / 5.0) * 5.0
        attn_weights = F.softmax(scores, dim=-1)
        
        out = torch.einsum('bhsS,bShv->bshv', attn_weights, v)
        out = out.reshape(B, S, -1)
        out = self.out_proj(out)
        return self.dropout(out)


class MLPBlock(nn.Module):
    """MLP块"""
    def __init__(self, embed_dim=1536):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, 2 * embed_dim)
        self.fc2 = nn.Linear(2 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x_norm = self.layer_norm(x)
        x_norm = self.fc1(x_norm)
        x_norm = F.relu(x_norm)
        x_norm = self.dropout(x_norm)
        x_norm = self.fc2(x_norm)
        return self.dropout(x_norm)


class TransformerTower(nn.Module):
    """Transformer塔"""
    def __init__(self, embed_dim=1536, num_layers=9):
        super().__init__()
        self.num_layers = num_layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(embed_dim) for _ in range(num_layers)
        ])
        self.mlp_layers = nn.ModuleList([
            MLPBlock(embed_dim) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for i in range(self.num_layers):
            x = x + self.attention_layers[i](x)
            x = x + self.mlp_layers[i](x)
        return x




class UpResBlock(nn.Module):
    """上采样残差块"""
    def __init__(self, embed_dim, out_channels, skip_channels):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.skip_channels = skip_channels
        
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.gelu1 = nn.GELU()
        self.conv1 = nn.Conv1d(embed_dim, out_channels, kernel_size=5, padding=2)
        
        self.layer_norm2 = nn.LayerNorm(out_channels)
        self.gelu2 = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        
        self.skip_conv = nn.Conv1d(skip_channels, out_channels, kernel_size=1)
        self.residual_scale = Parameter(torch.tensor(0.9))
        
    def forward(self, x, skip_connection):
        target_length = skip_connection.shape[-1]
        # 使用转置卷积进行上采样以匹配目标长度
        if x.shape[-1] != target_length:
            x_up = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
        else:
            x_up = x
        
        x_up_transposed = x_up.transpose(1, 2)
        x_norm = self.layer_norm1(x_up_transposed)
        x_norm = self.gelu1(x_norm)
        x_norm = x_norm.transpose(1, 2)
        
        out = self.conv1(x_norm)
        
        # 确保 skip_connection 的通道数与 skip_conv 的输入通道数匹配
        skip_out = self.skip_conv(skip_connection)
        
        # 确保 out 和 skip_out 的序列长度一致
        if out.shape[-1] != skip_out.shape[-1]:
            min_len = min(out.shape[-1], skip_out.shape[-1])
            out = out[..., :min_len]
            skip_out = skip_out[..., :min_len]
            
        out = out * self.residual_scale + skip_out
        
        out_transposed = out.transpose(1, 2)
        out_norm = self.layer_norm2(out_transposed)
        out_norm = self.gelu2(out_norm)
        out_norm = out_norm.transpose(1, 2)
        
        conv_out = self.conv2(out_norm)
        return out + conv_out


class SequenceDecoder(nn.Module):
    """序列解码器"""
    def __init__(self, embed_dim=1536):
        super().__init__()
        # 与 SequenceEncoder 的通道数对齐，顺序与 bin_sizes 匹配
        self.channels = [1536, 1408, 1280, 1152, 1024, 896, 768]
        self.upres_blocks = nn.ModuleList()
        current_dim = embed_dim
        for skip_channels, out_channels in zip(self.channels[:-1], self.channels[1:]):
            self.upres_blocks.append(UpResBlock(current_dim, out_channels, skip_channels))
            current_dim = out_channels
        self.final_conv = nn.Conv1d(current_dim, 4, kernel_size=1)

    def forward(self, x, intermediates):
        # bin_sizes 与 SequenceEncoder 的池化步骤对应，从高到低
        bin_sizes = [128, 64, 32, 16, 8, 4, 2]  # 从最高分辨率到最低
        for i, bin_size in enumerate(bin_sizes):
            skip = intermediates.get(f'bin_size_{bin_size}')
            if skip is not None and i < len(self.upres_blocks):
                # print(f"bin_size_{bin_size}: skip shape {skip.shape}, expected skip_channels {self.channels[i]}")
                if skip.shape[1] != self.channels[i]:
                    raise ValueError(
                        f"Channel mismatch at bin_size_{bin_size}: "
                        f"skip_connection has {skip.shape[1]} channels, "
                        f"but expected {self.channels[i]} channels"
                    )
                x = self.upres_blocks[i](x, skip)
        # 确保最终输出序列长度为 5000
        if x.shape[-1] != 5000:
            x = F.interpolate(x, size=5000, mode='linear', align_corners=False)
        return torch.sigmoid(self.final_conv(x))

        
   

class DualTaskPredictor(nn.Module):
    """基于AlphaGenome的双任务预测器"""
    def __init__(self, embed_dim=1536, hidden_dim=512):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.epi_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1)
        )
        self.eei_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1)
        )
        
    def forward(self, seq1_features, seq2_features):
        seq1_pooled = self.global_pool(seq1_features).squeeze(-1)
        seq2_pooled = self.global_pool(seq2_features).squeeze(-1)
        combined = torch.cat([seq1_pooled, seq2_pooled], dim=1)
        shared_features = self.feature_fusion(combined)
        epi_score = self.epi_head(shared_features)
        eei_score = self.eei_head(shared_features)
        return epi_score, eei_score


class AlphaGenomeCreate(nn.Module):
    """基于AlphaGenome架构的CREATE模型"""
    def __init__(self, seq_length=5000, embed_dim=1536):
        super().__init__()
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.seq1_encoder = SequenceEncoder(seq_length)
        self.seq2_encoder = SequenceEncoder(seq_length)
        self.seq1_proj = nn.Conv1d(1536, embed_dim, kernel_size=1)  # 匹配 SequenceEncoder 输出通道数
        self.seq2_proj = nn.Conv1d(1536, embed_dim, kernel_size=1)  # 匹配 SequenceEncoder 输出通道数
        self.transformer = TransformerTower(embed_dim, num_layers=6)
        self.seq1_decoder = SequenceDecoder(embed_dim)
        self.seq2_decoder = SequenceDecoder(embed_dim)
        self.dual_task_predictor = DualTaskPredictor(embed_dim, hidden_dim=512)
        
    def forward(self, seq1, seq2):
        seq1_encoded, seq1_intermediates = self.seq1_encoder(seq1)
        seq2_encoded, seq2_intermediates = self.seq2_encoder(seq2)
        seq1_features = self.seq1_proj(seq1_encoded)
        seq2_features = self.seq2_proj(seq2_encoded)
        seq1_features_t = seq1_features.transpose(1, 2)
        seq2_features_t = seq2_features.transpose(1, 2)
        seq1_transformed = self.transformer(seq1_features_t)
        seq2_transformed = self.transformer(seq2_features_t)
        seq1_transformed = seq1_transformed.transpose(1, 2)
        seq2_transformed = seq2_transformed.transpose(1, 2)
        seq1_recon = self.seq1_decoder(seq1_transformed, seq1_intermediates)
        seq2_recon = self.seq2_decoder(seq2_transformed, seq2_intermediates)
        epi_score, eei_score = self.dual_task_predictor(seq1_transformed, seq2_transformed)
        return {
            'epi_score': epi_score,
            'eei_score': eei_score,
            'seq1_recon': seq1_recon,
            'seq2_recon': seq2_recon
        }


def create(channel1=512, channel2=384, channel3=128, channel4=200, channel5=200, 
           embed_dim=1536, seq_length=5000):
    """创建基于AlphaGenome的CREATE模型"""
    return AlphaGenomeCreate(seq_length=seq_length, embed_dim=embed_dim)
