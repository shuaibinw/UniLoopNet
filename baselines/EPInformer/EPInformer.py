# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 21:00:50 2025

@author: 123
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, roc_curve, auc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import math
import copy
import warnings
warnings.filterwarnings('ignore')

# 设置GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class seq_256bp_encoder(nn.Module):
    def __init__(self, base_size=4, out_dim=128, conv_dim=256):
        super(seq_256bp_encoder, self).__init__()
        self.conv_dim = conv_dim
        self.out_dim = out_dim
        self.base_size = base_size
        
        # 修改卷积核大小以适应更长的序列
        self.stem_conv = nn.Sequential(
            nn.Conv1d(in_channels=base_size, out_channels=self.conv_dim, kernel_size=15, stride=1, padding=7),
            nn.ELU(),
        )
        
        self.conv_tower = nn.ModuleList([])
        conv_dim = [self.conv_dim, 128, 64, 64, 128]
        for i in range(4):
            self.conv_tower.append(nn.Sequential(
                nn.Conv1d(in_channels=conv_dim[i], out_channels=conv_dim[i+1], kernel_size=3, padding=1),
                nn.BatchNorm1d(conv_dim[i+1]),
                nn.ELU(),                   
                nn.MaxPool1d(kernel_size=2, stride=2),
            ))
            self.conv_tower.append(nn.Sequential(
                nn.Conv1d(in_channels=conv_dim[i+1], out_channels=conv_dim[i+1], kernel_size=1),
                nn.ELU(),
            ))
        
    def forward(self, seq_input):
        # seq_input shape: (batch_size, 4, 5000)
        x = self.stem_conv(seq_input)
        for i in range(0, len(self.conv_tower), 2):
            x = self.conv_tower[i](x)
            x = self.conv_tower[i+1](x) + x
        return x

class MHAttention_encoderLayer(nn.Module):
    def __init__(self, d_model=128, nhead=8, dropout=0.):
        super(MHAttention_encoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model)
        )
    
    def _sa_block(self, x, key_padding_mask, attn_mask):
        x, w = self.self_attn(x, x, x,
                           key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return x, w
        
    def forward(self, x, key_padding_mask=None, attn_mask=None):
        x2 = self.norm1(x)
        x2, attention_w = self._sa_block(x2, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x2 + x
        x2 = self.norm2(x)
        x = x + self.ff(x2)
        return x, attention_w

class MultiTaskEPInformer(nn.Module):
    def __init__(self, base_size=4, n_encoder=3, out_dim=128, head=4, device='cuda'):
        super(MultiTaskEPInformer, self).__init__()
        self.out_dim = out_dim
        self.n_encoder = n_encoder
        self.device = device
        
        # 共享的序列编码器
        self.seq_encoder = seq_256bp_encoder(base_size=base_size, out_dim=out_dim)
        
        # 共享的注意力层
        self.attn_encoder = get_clones(MHAttention_encoderLayer(d_model=out_dim, nhead=head), self.n_encoder)
        
        # 将卷积输出转换为固定维度
        self.conv_out = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 自适应池化到固定长度
            nn.Flatten(),
            nn.Linear(out_dim, out_dim),
            nn.ELU(),
        )
        
        # 任务特定的头部
        self.eei_head = nn.Sequential(
            nn.Linear(out_dim * 2, 256),  # 连接两个序列的特征
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.epi_head = nn.Sequential(
            nn.Linear(out_dim * 2, 256),  # 连接两个序列的特征
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def encode_sequence(self, seq):
        """编码单个序列"""
        # seq shape: (batch_size, 4, 5000)
        seq_embed = self.seq_encoder(seq)  # (batch_size, 128, seq_len_after_conv)
        seq_embed = self.conv_out(seq_embed)  # (batch_size, 128)
        return seq_embed
        
    def forward(self, seq1, seq2, task='eei'):
        """
        Args:
            seq1: (batch_size, 4, 5000)
            seq2: (batch_size, 4, 5000)
            task: 'eei' or 'epi'
        """
        # 编码两个序列
        seq1_embed = self.encode_sequence(seq1)  # (batch_size, 128)
        seq2_embed = self.encode_sequence(seq2)  # (batch_size, 128)
        
        # 连接两个序列的特征
        combined_embed = torch.cat([seq1_embed, seq2_embed], dim=1)  # (batch_size, 256)
        
        # 根据任务选择相应的头部
        if task == 'eei':
            output = self.eei_head(combined_embed)
        elif task == 'epi':
            output = self.epi_head(combined_embed)
        else:
            raise ValueError(f"Unknown task: {task}")
            
        return output.squeeze(-1)

class MultiTaskTrainer:
    def __init__(self, model, task_loaders, device, learning_rate=0.0001):
        self.model = model
        self.task_loaders = task_loaders
        self.device = device
        
        # 优化器
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 损失函数
        self.criterion = nn.BCELoss()
        
        # 记录训练历史
        self.history = {
            'eei': {'train_loss': [], 'val_loss': [], 'val_acc': []},
            'epi': {'train_loss': [], 'val_loss': [], 'val_acc': []}
        }
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        epoch_losses = {'eei': [], 'epi': []}
        
        # 获取每个任务的迭代器
        task_iterators = {}
        for task_name in ['eei', 'epi']:
            task_iterators[task_name] = iter(self.task_loaders[task_name]['train'])
        
        # 计算每个任务的batch数量
        task_batch_counts = {}
        for task_name in ['eei', 'epi']:
            task_batch_counts[task_name] = len(self.task_loaders[task_name]['train'])
        
        max_batches = max(task_batch_counts.values())
        
        for batch_idx in range(max_batches):
            self.optimizer.zero_grad()
            total_loss = 0
            
            # 对每个任务进行训练
            for task_name in ['eei', 'epi']:
                try:
                    seq1, seq2, labels = next(task_iterators[task_name])
                except StopIteration:
                    # 如果某个任务的数据用完了，重新开始
                    task_iterators[task_name] = iter(self.task_loaders[task_name]['train'])
                    seq1, seq2, labels = next(task_iterators[task_name])
                
                seq1, seq2, labels = seq1.to(self.device), seq2.to(self.device), labels.to(self.device)
                
                # 前向传播
                outputs = self.model(seq1, seq2, task=task_name)
                loss = self.criterion(outputs, labels.squeeze())
                
                # 累加损失
                total_loss += loss
                epoch_losses[task_name].append(loss.item())
            
            # 反向传播
            total_loss.backward()
            self.optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{max_batches}, '
                      f'EEI Loss: {np.mean(epoch_losses["eei"][-10:]):.4f}, '
                      f'EPI Loss: {np.mean(epoch_losses["epi"][-10:]):.4f}')
        
        # 记录训练损失
        for task_name in ['eei', 'epi']:
            self.history[task_name]['train_loss'].append(np.mean(epoch_losses[task_name]))
    
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        val_results = {}
        
        with torch.no_grad():
            for task_name in ['eei', 'epi']:
                val_losses = []
                val_preds = []
                val_true = []
                
                for seq1, seq2, labels in self.task_loaders[task_name]['val']:
                    seq1, seq2, labels = seq1.to(self.device), seq2.to(self.device), labels.to(self.device)
                    
                    outputs = self.model(seq1, seq2, task=task_name)
                    loss = self.criterion(outputs, labels.squeeze())
                    
                    val_losses.append(loss.item())
                    val_preds.extend(outputs.cpu().numpy())
                    val_true.extend(labels.squeeze().cpu().numpy())
                
                # 计算指标
                val_pred_labels = (np.array(val_preds) > 0.5).astype(int)
                val_acc = accuracy_score(val_true, val_pred_labels)
                val_f1 = f1_score(val_true, val_pred_labels)
                
                val_results[task_name] = {
                    'loss': np.mean(val_losses),
                    'acc': val_acc,
                    'f1': val_f1
                }
                
                # 记录验证结果
                self.history[task_name]['val_loss'].append(val_results[task_name]['loss'])
                self.history[task_name]['val_acc'].append(val_results[task_name]['acc'])
        
        # 打印验证结果
        print(f'Epoch {epoch} Validation:')
        for task_name in ['eei', 'epi']:
            print(f'  {task_name.upper()}: Loss={val_results[task_name]["loss"]:.4f}, '
                  f'Acc={val_results[task_name]["acc"]:.4f}, F1={val_results[task_name]["f1"]:.4f}')
        
        return val_results
    
    def train(self, epochs=300):
        """完整训练过程"""
        best_scores = {'eei': 0, 'epi': 0}
        
        for epoch in range(epochs):
            print(f'\n{"="*60}')
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'{"="*60}')
            
            # 训练
            self.train_epoch(epoch)
            
            # 验证
            val_results = self.validate_epoch(epoch)
            
            # 保存最佳模型
            for task_name in ['eei', 'epi']:
                if val_results[task_name]['acc'] > best_scores[task_name]:
                    best_scores[task_name] = val_results[task_name]['acc']
                    torch.save(self.model.state_dict(), f'best_model_{task_name}.pth')
                    print(f'  保存{task_name.upper()}最佳模型，验证准确率: {val_results[task_name]["acc"]:.4f}')

def test_model(model, task_loaders, device):
    """测试模型"""
    model.eval()
    test_results = {}
    
    # 创建输出目录
    os.makedirs('MultiTask_Results', exist_ok=True)
    
    with torch.no_grad():
        for task_name in ['eei', 'epi']:
            print(f'\n测试 {task_name.upper()} 任务...')
            
            test_preds = []
            test_true = []
            
            for seq1, seq2, labels in task_loaders[task_name]['test']:
                seq1, seq2, labels = seq1.to(device), seq2.to(device), labels.to(device)
                
                outputs = model(seq1, seq2, task=task_name)
                
                test_preds.extend(outputs.cpu().numpy())
                test_true.extend(labels.squeeze().cpu().numpy())
            
            test_preds = np.array(test_preds)
            test_true = np.array(test_true)
            test_pred_labels = (test_preds > 0.5).astype(int)
            
            # 计算评估指标
            from sklearn.metrics import precision_score, recall_score, precision_recall_curve
            
            accuracy = accuracy_score(test_true, test_pred_labels)
            f1_value = f1_score(test_true, test_pred_labels)
            precision_value = precision_score(test_true, test_pred_labels)
            recall_value = recall_score(test_true, test_pred_labels)
            auc_value = roc_auc_score(test_true, test_preds)
            aupr = average_precision_score(test_true, test_preds)
            
            test_results[task_name] = {
                'accuracy': accuracy,
                'f1': f1_value,
                'precision': precision_value,
                'recall': recall_value,
                'auc': auc_value,
                'aupr': aupr
            }
            
            # 打印结果
            print(f"{task_name.upper()} Test Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1 Score: {f1_value:.4f}")
            print(f"  Precision: {precision_value:.4f}")
            print(f"  Recall: {recall_value:.4f}")
            print(f"  AUC: {auc_value:.4f}")
            print(f"  AUPR: {aupr:.4f}")
            
            # 计算并保存ROC曲线数据
            fpr, tpr, roc_thresholds = roc_curve(test_true, test_preds)
            roc_auc = auc(fpr, tpr)
            
            # 保存ROC曲线数据
            roc_df = pd.DataFrame({
                'FPR': fpr,
                'TPR': tpr
            })
            roc_df.to_csv(f'MultiTask_Results/{task_name}_ROC.csv', index=False)
            
            # 计算并保存PR曲线数据
            precision_curve, recall_curve, pr_thresholds = precision_recall_curve(test_true, test_preds)
            
            pr_df = pd.DataFrame({
                'Recall': recall_curve,
                'Precision': precision_curve
            })
            pr_df.to_csv(f'MultiTask_Results/{task_name}_PRC.csv', index=False)
            
            # 保存评估指标
            with open(f'MultiTask_Results/{task_name}_metrics.txt', 'w') as f:
                f.write(f"{task_name.upper()} Model Evaluation Metrics\n")
                f.write(f"======================\n\n")
                f.write(f"Accuracy: {accuracy:.6f}\n")
                f.write(f"F1 Score: {f1_value:.6f}\n")
                f.write(f"Precision: {precision_value:.6f}\n")
                f.write(f"Recall: {recall_value:.6f}\n")
                f.write(f"AUC: {auc_value:.6f}\n")
                f.write(f"AUPR: {aupr:.6f}\n")
    
    return test_results

# 主程序
if __name__ == "__main__":
    # 导入数据加载器
    from data_loader import create_multitask_data_loaders, debug_data_shapes
    
    # 数据路径配置（请根据您的实际路径修改）
    data_paths = {
        # EPI任务数据路径
        'epi_train_seq1': '/public/home/shenyin_wsb_2606/Third/EP1000/5000/EPL_B.npz',
        'epi_train_seq2': '/public/home/shenyin_wsb_2606/Third/EP1000/5000/EPR_B.npz', 
        'epi_test_seq1': '/public/home/shenyin_wsb_2606/Third/EP1000/5000/EPL_C.npz',
        'epi_test_seq2': '/public/home/shenyin_wsb_2606/Third/EP1000/5000/EPR_C.npz',
        
        # 数据路径 - EEI任务 (新增)
        'eei_train_seq1': '/public/home/shenyin_wsb_2606/Third/EE1000/5000/EEL_B.npz',
        'eei_train_seq2': '/public/home/shenyin_wsb_2606/Third/EE1000/5000/EER_B.npz',
        'eei_test_seq1': '/public/home/shenyin_wsb_2606/Third/EE1000/5000/EEL_C.npz',
        'eei_test_seq2': '/public/home/shenyin_wsb_2606/Third/EE1000/5000/EER_C.npz',
    }
    
    # 创建数据加载器
    print("创建多任务数据加载器...")
    task_loaders, task_val_labels, task_test_labels = create_multitask_data_loaders(
        # EPI任务数据路径
        epi_train_seq1_path=data_paths['epi_train_seq1'],
        epi_train_seq2_path=data_paths['epi_train_seq2'],
        epi_test_seq1_path=data_paths['epi_test_seq1'],
        epi_test_seq2_path=data_paths['epi_test_seq2'],
        
        # EEI任务数据路径
        eei_train_seq1_path=data_paths['eei_train_seq1'],
        eei_train_seq2_path=data_paths['eei_train_seq2'],
        eei_test_seq1_path=data_paths['eei_test_seq1'],
        eei_test_seq2_path=data_paths['eei_test_seq2'],
        
        batch_size=32,
        val_ratio=0.1,
        random_state=42
    )
    
    # 调试数据形状
    debug_data_shapes(task_loaders)
    
    # 创建模型
    print("创建多任务模型...")
    model = MultiTaskEPInformer(
        base_size=4,
        n_encoder=3,
        out_dim=128,
        head=4,
        device=device
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 创建训练器
    trainer = MultiTaskTrainer(model, task_loaders, device, learning_rate=0.0001)
    
    # 训练模型
    print("开始训练...")
    trainer.train(epochs=50)
    
    # 测试模型
    print("开始测试...")
    
    # 加载最佳模型进行测试
    for task_name in ['eei', 'epi']:
        print(f'\n使用最佳{task_name.upper()}模型进行测试...')
        model.load_state_dict(torch.load(f'best_model_{task_name}.pth'))
        test_results = test_model(model, task_loaders, device)