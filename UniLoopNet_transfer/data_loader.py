#!/usr/bin/env python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class EnhancerDataset(Dataset):
    """增强子预测数据集（单输入）"""
    def __init__(self, seq_data, labels, dataset_name):
        self.seq_data = seq_data
        self.labels = labels
        self.dataset_name = dataset_name
        
        print(f"{dataset_name}数据集初始化:")
        print(f"  seq shape: {seq_data.shape}")
        print(f"  labels shape: {labels.shape}")
        print(f"  正样本比例: {np.mean(labels):.3f}")
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # 获取原始数据
        seq_raw = self.seq_data[idx]  # shape: (8000,)
        
        # 将8000长度的向量重塑为 (2000, 4) 然后转置为 (4, 2000)
        seq = torch.FloatTensor(seq_raw.reshape(2000, 4)).transpose(0, 1)  # (4, 2000)
        label = torch.FloatTensor([self.labels[idx]])
        
        return seq, label


def load_enhancer_data(seq_path, dataset_name):
    """加载增强子数据"""
    print(f"加载{dataset_name}数据: {seq_path}")
    
    # 加载序列数据
    seq_data = np.load(seq_path)
    sequences = seq_data['sequence']  # shape: (n_samples, 8000)
    labels = seq_data['label']       # shape: (n_samples,)
    
    # 验证数据可以正确重塑
    try:
        test_reshape = sequences[0].reshape(2000, 4)
        print(f"  - 重塑测试成功: {sequences[0].shape} -> {test_reshape.shape}")
    except Exception as e:
        print(f"  - 重塑测试失败: {e}")
        print(f"  - 8000是否能被4整除: {8000 % 4 == 0}")
        print(f"  - 8000/4 = {8000 // 4}")
    
    print(f"{dataset_name}数据加载完成:")
    print(f"  - 序列形状: {sequences.shape}")
    print(f"  - 标签形状: {labels.shape}")
    print(f"  - 正样本数量: {np.sum(labels)}")
    print(f"  - 负样本数量: {len(labels) - np.sum(labels)}")
    
    return sequences, labels


def create_enhancer_data_loaders(
    # 增强子预测数据路径
    train_seq_path,
    test_seq_path,
    
    batch_size=32, val_ratio=0.1, random_state=42
):
    """创建增强子预测数据加载器"""
    
    print(f"\n{'='*50}")
    print(f"处理增强子预测任务")
    print(f"{'='*50}")
    
    # 加载训练数据
    print(f"加载增强子训练数据...")
    train_seqs, train_labels = load_enhancer_data(train_seq_path, "训练集")
    
    # 加载测试数据
    print(f"加载增强子测试数据...")
    test_seqs, test_labels = load_enhancer_data(test_seq_path, "测试集")
    
    # 划分训练集和验证集
    print(f"按 {1-val_ratio}:{val_ratio} 划分训练集和验证集...")
    indices = np.arange(len(train_labels))
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=val_ratio, 
        random_state=random_state,
        stratify=train_labels
    )
    
    # 训练集
    train_seq = train_seqs[train_idx]
    train_lab = train_labels[train_idx]
    
    # 验证集
    val_seq = train_seqs[val_idx]
    val_lab = train_labels[val_idx]
    
    print(f"数据划分完成:")
    print(f"  - 训练集大小: {len(train_lab)} (正样本: {np.sum(train_lab)})")
    print(f"  - 验证集大小: {len(val_lab)} (正样本: {np.sum(val_lab)})")
    print(f"  - 测试集大小: {len(test_labels)} (正样本: {np.sum(test_labels)})")
    
    # 创建数据集
    train_dataset = EnhancerDataset(train_seq, train_lab, "训练集")
    val_dataset = EnhancerDataset(val_seq, val_lab, "验证集")
    test_dataset = EnhancerDataset(test_seqs, test_labels, "测试集")
    
    # 创建数据加载器
    data_loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    }
    
    return data_loaders, val_lab, test_labels


def debug_data_shapes(data_loaders):
    """调试数据形状"""
    print(f"\n{'='*50}")
    print("调试数据形状")
    print(f"{'='*50}")
    
    for split_name, loader in data_loaders.items():
        print(f"\n{split_name}:")
        try:
            for batch_idx, (seq, labels) in enumerate(loader):
                print(f"    Batch {batch_idx}:")
                print(f"      seq shape: {seq.shape}")  # 应该是 (batch_size, 4, 2000)
                print(f"      labels shape: {labels.shape}")
                print(f"      seq 数据范围: [{seq.min():.3f}, {seq.max():.3f}]")
                
                if batch_idx == 0:  # 只检查第一个batch
                    break
        except Exception as e:
            print(f"    调试时出错: {e}")
            import traceback
            traceback.print_exc()


def verify_data_format():
    """验证数据格式是否正确"""
    print("验证数据格式...")
    
    # 模拟8000长度的数据
    sample_seq = np.random.rand(8000)
    
    print(f"原始序列形状: {sample_seq.shape}")
    print(f"原始序列前20个值: {sample_seq[:20]}")
    
    # 尝试重塑
    reshaped = sample_seq.reshape(2000, 4)
    print(f"重塑后形状: {reshaped.shape}")
    print(f"重塑后前5行:")
    print(reshaped[:5])
    
    # 转置后的形状
    transposed = reshaped.T
    print(f"转置后形状: {transposed.shape}")


if __name__ == "__main__":
    # 首先验证数据格式
    verify_data_format()