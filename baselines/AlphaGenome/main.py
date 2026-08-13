#!/usr/bin/env python
import os
import torch
import numpy as np
from data_loader import create_multitask_data_loaders, debug_data_shapes
from layer import create
from create_train import CREATE_multitask_train

def main():
    # ========== 配置参数 ==========
    config = {
        # 数据路径 - EPI任务 (原有数据)
        'epi_train_seq1_path': '/public/home/shenyin_wsb_2606/Third/EP1000/5000/EPL_B.npz',
        'epi_train_seq2_path': '/public/home/shenyin_wsb_2606/Third/EP1000/5000/EPR_B.npz', 
        'epi_test_seq1_path': '/public/home/shenyin_wsb_2606/Third/EP1000/5000/EPL_C.npz',
        'epi_test_seq2_path': '/public/home/shenyin_wsb_2606/Third/EP1000/5000/EPR_C.npz',
        
        # 数据路径 - EEI任务 (新增)
        'eei_train_seq1_path': '/public/home/shenyin_wsb_2606/Third/EE1000/5000/EEL_B.npz',
        'eei_train_seq2_path': '/public/home/shenyin_wsb_2606/Third/EE1000/5000/EER_B.npz',
        'eei_test_seq1_path': '/public/home/shenyin_wsb_2606/Third/EE1000/5000/EEL_C.npz',
        'eei_test_seq2_path': '/public/home/shenyin_wsb_2606/Third/EE1000/5000/EER_C.npz',
        
        # 训练参数 - 针对AlphaGenome调整
        'batch_size': 8,  # 减小batch size，因为模型更大
        'val_ratio': 0.1,
        'lr': 1e-5,  # 降低学习率
        'max_epoch': 200,  # 减少训练轮数
        'pre_epoch': 30,   # 减少预训练轮数
        'seq_loss_weight': 0.5,  # 降低重构损失权重
        
        # 多任务权重
        'task_weights': {'epi': 1.0, 'eei': 1.0},
        
        # AlphaGenome模型参数
        'embed_dim': 1536,  # AlphaGenome标准嵌入维度
        'seq_length': 5000,
        
        # 其他
        'device': 'cuda:1' if torch.cuda.is_available() else 'cpu',
        'outdir': './results/',
        'random_state': 42
    }
    
    print("🚀 开始基于AlphaGenome的EPI+EEI双任务训练")
    print(f"使用设备: {config['device']}")
    print(f"序列长度: {config['seq_length']}")
    print(f"嵌入维度: {config['embed_dim']}")
    
    # ========== 数据加载 ==========
    print("\n📊 加载EPI+EEI双任务数据...")
    task_loaders, task_val_labels, task_test_labels = create_multitask_data_loaders(
        # EPI任务数据路径
        epi_train_seq1_path=config['epi_train_seq1_path'],
        epi_train_seq2_path=config['epi_train_seq2_path'],
        epi_test_seq1_path=config['epi_test_seq1_path'],
        epi_test_seq2_path=config['epi_test_seq2_path'],
        
        # EEI任务数据路径
        eei_train_seq1_path=config['eei_train_seq1_path'],
        eei_train_seq2_path=config['eei_train_seq2_path'],
        eei_test_seq1_path=config['eei_test_seq1_path'],
        eei_test_seq2_path=config['eei_test_seq2_path'],
        
        batch_size=config['batch_size'],
        val_ratio=config['val_ratio'],
        random_state=config['random_state']
    )
    
    # 调试数据形状
    debug_data_shapes(task_loaders)
    
    # ========== 模型创建 ==========
    print("\n🧠 创建基于AlphaGenome的EPI+EEI双任务模型...")
    model = create(
        embed_dim=config['embed_dim'],
        seq_length=config['seq_length']
    )
    
    # 计算模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"模型大小估计: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # ========== 开始训练 ==========
    print("\n🏋️ 开始基于AlphaGenome的EPI+EEI双任务训练...")
    CREATE_multitask_train(
        clf=model,
        train_loader=task_loaders,
        valid_loader=task_loaders,
        test_loader=task_loaders,
        valid_labels=task_val_labels,
        test_labels=task_test_labels,
        lr=config['lr'],
        max_epoch=config['max_epoch'],
        pre_epoch=config['pre_epoch'],
        seq_loss_weight=config['seq_loss_weight'],
        task_weights=config['task_weights'],
        outdir=config['outdir'],
        device=config['device']
    )


if __name__ == "__main__":
    main()