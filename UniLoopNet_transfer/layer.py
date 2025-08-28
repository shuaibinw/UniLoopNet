#!/usr/bin/env python
import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F


class SequenceEncoder(nn.Module):
    """单个序列编码器"""
    def __init__(self, channel1=256, channel2=256, channel3=128):
        super().__init__()
        self.channel1, self.channel2, self.channel3 = channel1, channel2, channel3
        
        # 第一层编码
        self.conv1 = torch.nn.Sequential(
            nn.Conv1d(4, out_channels=channel1, kernel_size=8, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=5, stride=4),
            nn.Dropout(0.1),
        )
        self.layernorm1 = nn.LayerNorm(channel1)
        
        # 第二层编码
        self.conv2 = torch.nn.Sequential(
            nn.Conv1d(channel1, out_channels=channel2, kernel_size=8, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(0.5),
        )
        
        # 第三层编码
        self.conv3 = torch.nn.Sequential(
            nn.Conv1d(channel2, out_channels=channel3, kernel_size=8, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        # x shape: (batch_size, 4, seq_length)
        out1 = self.conv1(x)
        out1 = self.layernorm1(out1.permute(0,2,1)).permute(0,2,1)
        
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        
        return out3


class SequenceDecoder(nn.Module):
    """单个序列解码器"""
    def __init__(self, embed_dim=128, channel4=300, channel5=300):
        super().__init__()
        
        # 解码器第一层
        self.convt3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear'),
            nn.ConvTranspose1d(embed_dim, channel4, kernel_size=9, stride=1),
            nn.ReLU(inplace=True),
        )
        
        # 解码器第二层
        self.convt2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='linear'),
            nn.ConvTranspose1d(channel4, channel5, kernel_size=9, stride=1),
            nn.ReLU(inplace=True),
        )
        
        # 解码器第三层，调整上采样因子以匹配目标长度2000
        self.convt1 = nn.Sequential(
            nn.Upsample(size=2000, mode='linear'),  # 直接指定输出长度为2000
            nn.ConvTranspose1d(channel5, 4, kernel_size=9, stride=1, padding=4),  # 添加padding以微调长度
            nn.BatchNorm1d(4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.convt3(x)
        out = self.convt2(out)
        out = self.convt1(out)
        return out


class EnhancerPredictor(nn.Module):
    """增强子预测模块（单输入）"""
    def __init__(self, embed_dim=128, hidden_dim=512):
        super().__init__()
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        
        # 增强子预测头
        self.enhancer_head = nn.Sequential(
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # 注意力机制（可选）
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)

    def forward(self, seq_features):
        # seq_features shape: (batch_size, embed_dim, seq_length)
        
        # 全局平均池化
        seq_pooled = self.global_pool(seq_features).squeeze(-1)  # (batch_size, embed_dim)
        
        # 特征提取
        features = self.feature_extractor(seq_pooled)  # (batch_size, hidden_dim//4)
        
        # 增强子预测
        enhancer_score = self.enhancer_head(features)  # (batch_size, 1)
        
        return enhancer_score


# CREATE for Enhancer Prediction (Single Input)
class create(nn.Module):
    def __init__(self, channel1=512, channel2=384, channel3=128, channel4=200, channel5=200, 
                 embed_dim=128):
        super().__init__()
        
        # 单个序列编码器
        self.seq_encoder = SequenceEncoder(channel1, channel2, channel3)
        
        # 特征转换层（替代量化器）
        self.feature_transform = nn.Conv1d(channel3, embed_dim, kernel_size=1, stride=1)
        
        # 序列解码器
        self.seq_decoder = SequenceDecoder(embed_dim, channel4, channel5)
        
        # 增强子预测器
        self.enhancer_predictor = EnhancerPredictor(embed_dim, hidden_dim=512)

    def forward(self, seq):
        # 编码序列
        seq_enc = self.seq_encoder(seq)
        
        # 特征转换（替代向量量化）
        seq_features = self.feature_transform(seq_enc)
        
        # 重构序列
        seq_recon = self.seq_decoder(seq_features)
        
        # 增强子预测
        enhancer_score = self.enhancer_predictor(seq_features)
        
        return {
            'enhancer_score': enhancer_score,
            'seq_recon': seq_recon
        }