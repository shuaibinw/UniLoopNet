#!/usr/bin/env python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class SingleTaskDataset(Dataset):
    """单任务数据集"""
    def __init__(self, seq1_data, seq2_data, labels, task_name):
        self.seq1_data = seq1_data
        self.seq2_data = seq2_data
        self.labels = labels
        self.task_name = task_name
        
        print(f"{task_name.upper()}任务数据集初始化:")
        print(f"  seq1 shape: {seq1_data.shape}")
        print(f"  seq2 shape: {seq2_data.shape}")
        print(f"  labels shape: {labels.shape}")
        print(f"  正样本比例: {np.mean(labels):.3f}")
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # 获取原始数据
        seq1_raw = self.seq1_data[idx]  # shape: (20000,)
        seq2_raw = self.seq2_data[idx]  # shape: (20000,)
        
        # 将20000长度的向量重塑为 (5000, 4) 然后转置为 (4, 5000)
        seq1 = torch.FloatTensor(seq1_raw.reshape(5000, 4)).transpose(0, 1)  # (4, 5000)
        seq2 = torch.FloatTensor(seq2_raw.reshape(5000, 4)).transpose(0, 1)  # (4, 5000)
        label = torch.FloatTensor([self.labels[idx]])
        
        return seq1, seq2, label


def load_task_data(seq1_path, seq2_path, task_name):
    """加载单个任务的数据"""
    print(f"加载{task_name.upper()}任务数据: {seq1_path}, {seq2_path}")
    
    # 加载第一个序列数据
    seq1_data = np.load(seq1_path)
    seq1_seqs = seq1_data['sequence']  # shape: (n_samples, 20000)
    seq1_labels = seq1_data['label']   # shape: (n_samples,)
    
    # 加载第二个序列数据
    seq2_data = np.load(seq2_path)
    seq2_seqs = seq2_data['sequence']  # shape: (n_samples, 20000)
    seq2_labels = seq2_data['label']   # shape: (n_samples,)
    
    # 确保同一任务内的两个序列文件样本数量一致
    assert len(seq1_seqs) == len(seq2_seqs), f"{task_name}任务的两个序列文件样本数量不一致"
    
    # 确保标签一致
    if not np.array_equal(seq1_labels, seq2_labels):
        print(f"警告: {task_name}任务的两个序列文件标签不完全一致，使用第一个文件的标签")
    
    # 验证数据可以正确重塑
    try:
        test_reshape = seq1_seqs[0].reshape(5000, 4)
        print(f"  - 重塑测试成功: {seq1_seqs[0].shape} -> {test_reshape.shape}")
    except Exception as e:
        print(f"  - 重塑测试失败: {e}")
        print(f"  - 20000是否能被4整除: {20000 % 4 == 0}")
        print(f"  - 20000/4 = {20000 // 4}")
    
    print(f"{task_name.upper()}数据加载完成:")
    print(f"  - 序列1形状: {seq1_seqs.shape}")
    print(f"  - 序列2形状: {seq2_seqs.shape}")
    print(f"  - 标签形状: {seq1_labels.shape}")
    print(f"  - 正样本数量: {np.sum(seq1_labels)}")
    print(f"  - 负样本数量: {len(seq1_labels) - np.sum(seq1_labels)}")
    
    return seq1_seqs, seq2_seqs, seq1_labels


def create_multitask_data_loaders(
    # EPI任务数据路径
    epi_train_seq1_path, epi_train_seq2_path,
    epi_test_seq1_path, epi_test_seq2_path,
    
    # EEI任务数据路径
    eei_train_seq1_path, eei_train_seq2_path,
    eei_test_seq1_path, eei_test_seq2_path,
    
    batch_size=32, val_ratio=0.1, random_state=42
):
    """创建EPI+EEI双任务数据加载器"""
    
    task_loaders = {}
    task_val_labels = {}
    task_test_labels = {}
    
    # 任务配置 - 只保留EPI和EEI
    tasks = {
        'epi': {
            'train_seq1': epi_train_seq1_path,
            'train_seq2': epi_train_seq2_path,
            'test_seq1': epi_test_seq1_path,
            'test_seq2': epi_test_seq2_path
        },
        'eei': {
            'train_seq1': eei_train_seq1_path,
            'train_seq2': eei_train_seq2_path,
            'test_seq1': eei_test_seq1_path,
            'test_seq2': eei_test_seq2_path
        }
    }
    
    for task_name, paths in tasks.items():
        print(f"\n{'='*50}")
        print(f"处理 {task_name.upper()} 任务")
        print(f"{'='*50}")
        
        # 加载训练数据
        print(f"加载{task_name.upper()}训练数据...")
        train_seq1, train_seq2, train_labels = load_task_data(
            paths['train_seq1'], paths['train_seq2'], task_name
        )
        
        # 加载测试数据
        print(f"加载{task_name.upper()}测试数据...")
        test_seq1, test_seq2, test_labels = load_task_data(
            paths['test_seq1'], paths['test_seq2'], task_name
        )
        
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
        train_s1 = train_seq1[train_idx]
        train_s2 = train_seq2[train_idx]
        train_lab = train_labels[train_idx]
        
        # 验证集
        val_s1 = train_seq1[val_idx]
        val_s2 = train_seq2[val_idx]
        val_lab = train_labels[val_idx]
        
        print(f"数据划分完成:")
        print(f"  - 训练集大小: {len(train_lab)} (正样本: {np.sum(train_lab)})")
        print(f"  - 验证集大小: {len(val_lab)} (正样本: {np.sum(val_lab)})")
        print(f"  - 测试集大小: {len(test_labels)} (正样本: {np.sum(test_labels)})")
        
        # 创建数据集
        train_dataset = SingleTaskDataset(train_s1, train_s2, train_lab, task_name)
        val_dataset = SingleTaskDataset(val_s1, val_s2, val_lab, task_name)
        test_dataset = SingleTaskDataset(test_seq1, test_seq2, test_labels, task_name)
        
        # 创建数据加载器
        task_loaders[task_name] = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
            'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        }
        
        # 保存标签用于评估
        task_val_labels[task_name] = val_lab
        task_test_labels[task_name] = test_labels
    
    return task_loaders, task_val_labels, task_test_labels


def debug_data_shapes(task_loaders):
    """调试数据形状"""
    print(f"\n{'='*50}")
    print("调试数据形状")
    print(f"{'='*50}")
    
    for task_name, loaders in task_loaders.items():
        print(f"\n{task_name.upper()}任务:")
        
        for split_name, loader in loaders.items():
            print(f"  {split_name}:")
            try:
                for batch_idx, (seq1, seq2, labels) in enumerate(loader):
                    print(f"    Batch {batch_idx}:")
                    print(f"      seq1 shape: {seq1.shape}")  # 应该是 (batch_size, 4, 5000)
                    print(f"      seq2 shape: {seq2.shape}")  # 应该是 (batch_size, 4, 5000)
                    print(f"      labels shape: {labels.shape}")
                    print(f"      seq1 数据范围: [{seq1.min():.3f}, {seq1.max():.3f}]")
                    print(f"      seq2 数据范围: [{seq2.min():.3f}, {seq2.max():.3f}]")
                    
                    if batch_idx == 0:  # 只检查第一个batch
                        break
            except Exception as e:
                print(f"    调试时出错: {e}")
                import traceback
                traceback.print_exc()


def verify_data_format():
    """验证数据格式是否正确"""
    print("验证数据格式...")
    
    # 模拟20000长度的数据
    sample_seq = np.random.rand(20000)
    
    print(f"原始序列形状: {sample_seq.shape}")
    print(f"原始序列前20个值: {sample_seq[:20]}")
    
    # 尝试重塑
    reshaped = sample_seq.reshape(5000, 4)
    print(f"重塑后形状: {reshaped.shape}")
    print(f"重塑后前5行:")
    print(reshaped[:5])
    
    # 转置后的形状
    transposed = reshaped.T
    print(f"转置后形状: {transposed.shape}")


if __name__ == "__main__":
    # 首先验证数据格式
    verify_data_format()