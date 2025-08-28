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
        seq1_raw = self.seq1_data[idx]  # shape: (20000,) for EPI/EEI or (20000,) for padded CTCF
        seq2_raw = self.seq2_data[idx]  # shape: (20000,) for EPI/EEI or (20000,) for padded CTCF
        
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
    seq1_seqs = seq1_data['sequence']  # shape: (n_samples, 4000) for CTCF or (n_samples, 20000) for others
    seq1_labels = seq1_data['label']   # shape: (n_samples,)
    
    # 加载第二个序列数据
    seq2_data = np.load(seq2_path)
    seq2_seqs = seq2_data['sequence']  # shape: (n_samples, 4000) for CTCF or (n_samples, 20000) for others
    seq2_labels = seq2_data['label']   # shape: (n_samples,)
    
    # 确保同一任务内的两个序列文件样本数量一致
    assert len(seq1_seqs) == len(seq2_seqs), f"{task_name}任务的两个序列文件样本数量不一致"
    
    # 确保标签一致
    if not np.array_equal(seq1_labels, seq2_labels):
        print(f"警告: {task_name}任务的两个序列文件标签不完全一致，使用第一个文件的标签")
    
    print(f"{task_name.upper()}数据加载完成:")
    print(f"  - 序列1形状: {seq1_seqs.shape}")
    print(f"  - 序列2形状: {seq2_seqs.shape}")
    print(f"  - 标签形状: {seq1_labels.shape}")
    print(f"  - 正样本数量: {np.sum(seq1_labels)}")
    print(f"  - 负样本数量: {len(seq1_labels) - np.sum(seq1_labels)}")
    
    return seq1_seqs, seq2_seqs, seq1_labels


# def pad_ctcf_sequences(sequences, target_length=5000, original_length=1000, padding_strategy='symmetric'):
#     """
#     将CTCF序列从1000x4填充到5000x4
    
#     Args:
#         sequences: 输入序列，shape为(n_samples, 4000) (1000*4展平)
#         target_length: 目标长度 (5000)
#         original_length: 原始长度 (1000)
#         padding_strategy: 填充策略
#             - 'symmetric': 对称填充
#             - 'zero': 零填充
#             - 'edge': 边界值填充
#             - 'reflect': 反射填充
    
#     Returns:
#         padded_sequences: 填充后的序列，shape为(n_samples, 20000) (5000*4展平)
#     """
#     n_samples = sequences.shape[0]
    
#     # 首先将展平的序列重塑为(n_samples, 1000, 4)
#     reshaped_seqs = sequences.reshape(n_samples, original_length, 4)
#     print(f"重塑CTCF序列: {sequences.shape} -> {reshaped_seqs.shape}")
    
#     # 计算需要填充的长度
#     pad_length = target_length - original_length  # 5000 - 1000 = 4000
#     pad_each_side = pad_length // 2  # 每边填充2000
#     pad_remainder = pad_length % 2   # 余数
    
#     print(f"填充策略: {padding_strategy}")
#     print(f"需要填充长度: {pad_length} (每边: {pad_each_side}, 余数: {pad_remainder})")
    
#     # 创建填充后的数组
#     padded_seqs = np.zeros((n_samples, target_length, 4), dtype=np.float32)
    
#     if padding_strategy == 'symmetric':
#         # 对称填充：在两端各填充一半长度
#         for i in range(n_samples):
#             seq = reshaped_seqs[i]  # (1000, 4)
            
#             # 中心位置放置原始序列
#             start_idx = pad_each_side
#             end_idx = start_idx + original_length
#             padded_seqs[i, start_idx:end_idx] = seq
            
#             # 左侧填充
#             if pad_each_side > 0:
#                 # 使用原始序列的反向来填充左侧
#                 left_pad_seq = seq[:pad_each_side][::-1]  # 取前pad_each_side个，然后反转
#                 padded_seqs[i, :pad_each_side] = left_pad_seq
            
#             # 右侧填充
#             right_pad_length = pad_each_side + pad_remainder
#             if right_pad_length > 0:
#                 # 使用原始序列的反向来填充右侧
#                 right_pad_seq = seq[-right_pad_length:][::-1]  # 取后right_pad_length个，然后反转
#                 padded_seqs[i, end_idx:] = right_pad_seq
                
                
def pad_ctcf_sequences(sequences, target_length=5000, original_length=1000, padding_strategy='symmetric'):
    n_samples = sequences.shape[0]
    
    # 重塑序列为 (n_samples, original_length, 4)
    reshaped_seqs = sequences.reshape(n_samples, original_length, 4)
    print(f"重塑CTCF序列: {sequences.shape} -> {reshaped_seqs.shape}")
    
    # 计算填充长度
    pad_length = target_length - original_length
    pad_each_side = pad_length // 2
    pad_remainder = pad_length % 2
    
    print(f"填充策略: {padding_strategy}")
    print(f"需要填充长度: {pad_length} (每边: {pad_each_side}, 余数: {pad_remainder})")
    
    # 创建填充后的数组
    padded_seqs = np.zeros((n_samples, target_length, 4), dtype=np.float32)
    
    if padding_strategy == 'symmetric':
        for i in range(n_samples):
            seq = reshaped_seqs[i]  # (1000, 4)
            
            # 中心位置放置原始序列
            start_idx = pad_each_side
            end_idx = start_idx + original_length
            padded_seqs[i, start_idx:end_idx] = seq
            
            # 左侧填充
            if pad_each_side > 0:
                # 使用原始序列的反向来填充左侧，重复序列以填满
                left_pad_seq = seq[::-1]  # 反转序列，形状 (1000, 4)
                # 计算需要重复的次数
                repeat_times = (pad_each_side + original_length - 1) // original_length
                # 重复序列
                extended_seq = np.tile(left_pad_seq, (repeat_times, 1))[:pad_each_side]
                padded_seqs[i, :pad_each_side] = extended_seq
            
            # 右侧填充
            right_pad_length = pad_each_side + pad_remainder
            if right_pad_length > 0:
                # 使用原始序列的反向来填充右侧，重复序列以填满
                right_pad_seq = seq[::-1]  # 反转序列，形状 (1000, 4)
                repeat_times = (right_pad_length + original_length - 1) // original_length
                extended_seq = np.tile(right_pad_seq, (repeat_times, 1))[:right_pad_length]
                padded_seqs[i, end_idx:] = extended_seq
    
  
                
    elif padding_strategy == 'zero':
        # 零填充：中心放置原始序列，两端填充0
        for i in range(n_samples):
            start_idx = pad_each_side
            end_idx = start_idx + original_length
            padded_seqs[i, start_idx:end_idx] = reshaped_seqs[i]
            
    elif padding_strategy == 'edge':
        # 边界值填充：使用序列的第一个和最后一个值
        for i in range(n_samples):
            seq = reshaped_seqs[i]
            start_idx = pad_each_side
            end_idx = start_idx + original_length
            
            # 中心放置原始序列
            padded_seqs[i, start_idx:end_idx] = seq
            
            # 左侧用第一个值填充
            if pad_each_side > 0:
                padded_seqs[i, :start_idx] = seq[0]
            
            # 右侧用最后一个值填充
            if pad_each_side + pad_remainder > 0:
                padded_seqs[i, end_idx:] = seq[-1]
                
    elif padding_strategy == 'reflect':
        # 反射填充
        for i in range(n_samples):
            seq = reshaped_seqs[i]
            start_idx = pad_each_side
            end_idx = start_idx + original_length
            
            # 中心放置原始序列
            padded_seqs[i, start_idx:end_idx] = seq
            
            # 左侧反射填充
            if pad_each_side > 0:
                reflect_length = min(pad_each_side, original_length)
                padded_seqs[i, start_idx-reflect_length:start_idx] = seq[:reflect_length][::-1]
                
                # 如果还需要更多填充，重复反射
                remaining = pad_each_side - reflect_length
                while remaining > 0:
                    fill_length = min(remaining, original_length)
                    start_fill = start_idx - reflect_length - fill_length
                    padded_seqs[i, start_fill:start_fill+fill_length] = seq[:fill_length]
                    remaining -= fill_length
            
            # 右侧反射填充
            right_pad_length = pad_each_side + pad_remainder
            if right_pad_length > 0:
                reflect_length = min(right_pad_length, original_length)
                padded_seqs[i, end_idx:end_idx+reflect_length] = seq[-reflect_length:][::-1]
                
                # 如果还需要更多填充，重复反射
                remaining = right_pad_length - reflect_length
                fill_start = end_idx + reflect_length
                while remaining > 0:
                    fill_length = min(remaining, original_length)
                    padded_seqs[i, fill_start:fill_start+fill_length] = seq[-fill_length:]
                    remaining -= fill_length
                    fill_start += fill_length
    
    # 将填充后的序列展平为(n_samples, 20000)
    flattened_padded = padded_seqs.reshape(n_samples, target_length * 4)
    
    print(f"填充完成: {reshaped_seqs.shape} -> {padded_seqs.shape} -> {flattened_padded.shape}")
    print(f"填充前数据范围: [{sequences.min():.3f}, {sequences.max():.3f}]")
    print(f"填充后数据范围: [{flattened_padded.min():.3f}, {flattened_padded.max():.3f}]")
    
    return flattened_padded


def load_ctcf_data(seq1_path, seq2_path, target_length=5000, original_length=1000, padding_strategy='symmetric'):
    """加载并填充CTCF数据"""
    print(f"加载CTCF任务数据: {seq1_path}, {seq2_path}")
    print(f"填充配置: {original_length} -> {target_length}, 策略: {padding_strategy}")
    
    # 加载原始CTCF数据
    seq1_data = np.load(seq1_path)
    seq1_seqs = seq1_data['sequence']  # shape: (n_samples, 4000)
    seq1_labels = seq1_data['label']   # shape: (n_samples,)
    
    seq2_data = np.load(seq2_path)
    seq2_seqs = seq2_data['sequence']  # shape: (n_samples, 4000)
    seq2_labels = seq2_data['label']   # shape: (n_samples,)
    
    print(f"原始CTCF数据加载完成:")
    print(f"  - 序列1形状: {seq1_seqs.shape}")
    print(f"  - 序列2形状: {seq2_seqs.shape}")
    print(f"  - 标签形状: {seq1_labels.shape}")
    
    # 验证数据长度
    expected_length = original_length * 4
    if seq1_seqs.shape[1] != expected_length:
        raise ValueError(f"CTCF序列长度不匹配: 期望 {expected_length}, 实际 {seq1_seqs.shape[1]}")
    
    # 填充序列
    print("开始填充CTCF序列...")
    seq1_padded = pad_ctcf_sequences(seq1_seqs, target_length, original_length, padding_strategy)
    seq2_padded = pad_ctcf_sequences(seq2_seqs, target_length, original_length, padding_strategy)
    
    # 确保标签一致
    if not np.array_equal(seq1_labels, seq2_labels):
        print("警告: CTCF任务的两个序列文件标签不完全一致，使用第一个文件的标签")
    
    print(f"CTCF数据处理完成:")
    print(f"  - 填充后序列1形状: {seq1_padded.shape}")
    print(f"  - 填充后序列2形状: {seq2_padded.shape}")
    print(f"  - 标签形状: {seq1_labels.shape}")
    print(f"  - 正样本数量: {np.sum(seq1_labels)}")
    print(f"  - 负样本数量: {len(seq1_labels) - np.sum(seq1_labels)}")
    
    return seq1_padded, seq2_padded, seq1_labels


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


def create_ctcf_data_loaders(
    ctcf_train_seq1_path,
    ctcf_train_seq2_path,
    ctcf_test_seq1_path,
    ctcf_test_seq2_path,
    sampling_ratios=[0.0, 0.02, 0.1, 0.5, 1.0],
    batch_size=32,
    val_ratio=0.1,
    random_state=42,
    target_length=5000,
    original_length=1000,
    padding_strategy='symmetric'
):
    """创建CTCF任务数据加载器，支持不同采样率和序列填充"""
    
    print(f"\n{'='*50}")
    print(f"处理 CTCF 任务")
    print(f"{'='*50}")
    
    # 加载训练数据
    print(f"加载CTCF训练数据...")
    train_seq1, train_seq2, train_labels = load_ctcf_data(
        ctcf_train_seq1_path, ctcf_train_seq2_path, 
        target_length, original_length, padding_strategy
    )
    
    # 加载测试数据
    print(f"加载CTCF测试数据...")
    test_seq1, test_seq2, test_labels = load_ctcf_data(
        ctcf_test_seq1_path, ctcf_test_seq2_path,
        target_length, original_length, padding_strategy
    )
    
    ctcf_loaders = {}
    
    for ratio in sampling_ratios:
        print(f"\n创建采样率 {ratio*100}% 的数据加载器...")
        
        if ratio == 0.0:
            # 0%采样率：不提供训练数据，只有测试数据
            ctcf_loaders[ratio] = {
                'train': None,
                'val': None,
                'test': DataLoader(
                    SingleTaskDataset(test_seq1, test_seq2, test_labels, 'ctcf'),
                    batch_size=batch_size, shuffle=False, num_workers=0
                ),
                'train_labels': np.array([]),
                'val_labels': np.array([]),
                'test_labels': test_labels
            }
        else:
            # 按比例采样训练数据
            n_samples = len(train_labels)
            indices = np.arange(n_samples)
            
            if ratio == 1.0:
                # 当采样率为 100% 时，直接使用所有数据
                sampled_seq1 = train_seq1
                sampled_seq2 = train_seq2
                sampled_labels = train_labels
                n_sampled = n_samples
            else:
                # 分层采样确保正负样本比例
                n_sampled = int(n_samples * ratio)
                sampled_idx, _ = train_test_split(
                    indices,
                    train_size=n_sampled,
                    random_state=random_state,
                    stratify=train_labels
                )
                sampled_seq1 = train_seq1[sampled_idx]
                sampled_seq2 = train_seq2[sampled_idx]
                sampled_labels = train_labels[sampled_idx]
            
            # 划分训练集和验证集
            train_idx, val_idx = train_test_split(
                np.arange(len(sampled_labels)),
                test_size=val_ratio,
                random_state=random_state,
                stratify=sampled_labels
            )
            
            # 最终的训练集和验证集
            final_train_s1 = sampled_seq1[train_idx]
            final_train_s2 = sampled_seq2[train_idx]
            final_train_lab = sampled_labels[train_idx]
            
            final_val_s1 = sampled_seq1[val_idx]
            final_val_s2 = sampled_seq2[val_idx]
            final_val_lab = sampled_labels[val_idx]
            
            print(f"  采样率 {ratio*100}% 数据统计:")
            print(f"    - 原始数据: {n_samples} 样本")
            print(f"    - 采样后: {n_sampled} 样本")
            print(f"    - 训练集: {len(final_train_lab)} 样本 (正样本: {np.sum(final_train_lab)})")
            print(f"    - 验证集: {len(final_val_lab)} 样本 (正样本: {np.sum(final_val_lab)})")
            
            # 创建数据加载器
            train_dataset = SingleTaskDataset(final_train_s1, final_train_s2, final_train_lab, 'ctcf')
            val_dataset = SingleTaskDataset(final_val_s1, final_val_s2, final_val_lab, 'ctcf')
            test_dataset = SingleTaskDataset(test_seq1, test_seq2, test_labels, 'ctcf')
            
            ctcf_loaders[ratio] = {
                'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
                'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
                'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
                'train_labels': final_train_lab,
                'val_labels': final_val_lab,
                'test_labels': test_labels
            }
    
    return ctcf_loaders


def debug_data_shapes(task_loaders):
    """调试数据形状"""
    print(f"\n{'='*50}")
    print("调试数据形状")
    print(f"{'='*50}")
    
    for task_name, loaders in task_loaders.items():
        print(f"\n{task_name.upper()}任务:")
        
        for split_name, loader in loaders.items():
            if loader is None:
                print(f"  {split_name}: None")
                continue
                
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


def verify_ctcf_padding():
    """验证CTCF填充是否正确"""
    print("验证CTCF填充功能...")
    
    # 创建模拟CTCF数据 (1000*4 = 4000)
    sample_ctcf = np.random.rand(2, 4000).astype(np.float32)
    print(f"模拟CTCF数据形状: {sample_ctcf.shape}")
    
    # 测试不同填充策略
    strategies = ['symmetric', 'zero', 'edge', 'reflect']
    
    for strategy in strategies:
        print(f"\n测试填充策略: {strategy}")
        padded = pad_ctcf_sequences(sample_ctcf, 5000, 1000, strategy)
        print(f"填充后形状: {padded.shape}")
        print(f"数据范围: [{padded.min():.3f}, {padded.max():.3f}]")
        
        # 验证可以正确重塑
        reshaped = padded.reshape(-1, 5000, 4)
        print(f"重塑验证: {padded.shape} -> {reshaped.shape}")


if __name__ == "__main__":
    # 验证CTCF填充功能
    verify_ctcf_padding()