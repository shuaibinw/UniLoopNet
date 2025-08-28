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
        
        # 解码器第三层
        self.convt1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='linear'),
            nn.ConvTranspose1d(channel5, 4, kernel_size=9, stride=1),
            nn.BatchNorm1d(4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.convt3(x)
        out = self.convt2(out)
        out = self.convt1(out)
        return out


class MultiTaskPredictor(nn.Module):
    """多任务预测模块：EPI、EEI和CTCF"""
    def __init__(self, embed_dim=128, hidden_dim=512):
        super().__init__()
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 共享特征提取层
        self.shared_fc = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        
        # 任务特定的预测头
        # EPI (Enhancer-Promoter Interaction)
        self.epi_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # EEI (Enhancer-Enhancer Interaction)
        self.eei_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # CTCF (持续学习新任务)
        self.ctcf_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # 注意力机制（可选）
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)

    def forward(self, seq1_features, seq2_features, task='all'):
        # seq1_features, seq2_features shape: (batch_size, embed_dim, seq_length)
        
        # 全局平均池化
        seq1_pooled = self.global_pool(seq1_features).squeeze(-1)  # (batch_size, embed_dim)
        seq2_pooled = self.global_pool(seq2_features).squeeze(-1)  # (batch_size, embed_dim)
        
        # 特征拼接
        combined = torch.cat([seq1_pooled, seq2_pooled], dim=1)  # (batch_size, embed_dim*2)
        
        # 共享特征提取
        shared_features = self.shared_fc(combined)  # (batch_size, hidden_dim//2)
        
        # 根据任务返回相应预测
        if task == 'all':
            epi_score = self.epi_head(shared_features)      # (batch_size, 1)
            eei_score = self.eei_head(shared_features)      # (batch_size, 1)
            ctcf_score = self.ctcf_head(shared_features)    # (batch_size, 1)
            return epi_score, eei_score, ctcf_score
        elif task == 'epi':
            return self.epi_head(shared_features)
        elif task == 'eei':
            return self.eei_head(shared_features)
        elif task == 'ctcf':
            return self.ctcf_head(shared_features)
        else:
            raise ValueError(f"Unknown task: {task}")


# CREATE for Multi-task Learning (EPI + EEI + CTCF) with Continual Learning
class create(nn.Module):
    def __init__(self, channel1=512, channel2=384, channel3=128, channel4=200, channel5=200, 
                 embed_dim=128):
        super().__init__()
        
        # 两个独立的序列编码器
        self.seq1_encoder = SequenceEncoder(channel1, channel2, channel3)
        self.seq2_encoder = SequenceEncoder(channel1, channel2, channel3)
        
        # 特征转换层（替代量化器）
        self.seq1_feature_transform = nn.Conv1d(channel3, embed_dim, kernel_size=1, stride=1)
        self.seq2_feature_transform = nn.Conv1d(channel3, embed_dim, kernel_size=1, stride=1)
        
        # 两个独立的解码器
        self.seq1_decoder = SequenceDecoder(embed_dim, channel4, channel5)
        self.seq2_decoder = SequenceDecoder(embed_dim, channel4, channel5)
        
        # 多任务预测器（包含CTCF）
        self.multi_task_predictor = MultiTaskPredictor(embed_dim, hidden_dim=512)

    def forward(self, seq1, seq2, task='all'):
        # 编码两个序列
        seq1_enc = self.seq1_encoder(seq1)
        seq2_enc = self.seq2_encoder(seq2)
        
        # 特征转换（替代向量量化）
        seq1_features = self.seq1_feature_transform(seq1_enc)
        seq2_features = self.seq2_feature_transform(seq2_enc)
        
        # 重构序列
        seq1_recon = self.seq1_decoder(seq1_features)
        seq2_recon = self.seq2_decoder(seq2_features)
        
        # 多任务预测
        if task == 'all':
            epi_score, eei_score, ctcf_score = self.multi_task_predictor(seq1_features, seq2_features, task='all')
            return {
                'epi_score': epi_score,
                'eei_score': eei_score,
                'ctcf_score': ctcf_score,
                'seq1_recon': seq1_recon,
                'seq2_recon': seq2_recon
            }
        else:
            task_score = self.multi_task_predictor(seq1_features, seq2_features, task=task)
            return {
                f'{task}_score': task_score,
                'seq1_recon': seq1_recon,
                'seq2_recon': seq2_recon
            }