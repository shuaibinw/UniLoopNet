#!/usr/bin/env python
import os
import torch
import numpy as np
from data_loader import create_enhancer_data_loaders, debug_data_shapes
from layer import create
from UniLoopNet_trasfer_train import CREATE_enhancer_train

def main():
    # ========== 配置参数 ==========
    config = {
        # 数据路径 - 增强子预测任务
        'train_seq_path': 'GM12878_train.npz',
        'test_seq_path': 'GM12878_test.npz',
        
        # 训练参数
        'batch_size': 24,
        'val_ratio': 0.1,
        'lr': 3e-5,
        'max_epoch': 300,
        'pre_epoch': 50,
        'seq_loss_weight': 1.0,
        
        # 模型参数
        'channel1': 512,
        'channel2': 384,
        'channel3': 128,
        'channel4': 200,
        'channel5': 200,
        'embed_dim': 128,
        'seq_length': 2000,  # 修改为2000以匹配输入序列长度
        
        # 其他
        'device': 'cuda:1' if torch.cuda.is_available() else 'cpu',
        'outdir': './results/',
        'random_state': 42
    }
    
    print("🚀 开始增强子预测训练")
    print(f"使用设备: {config['device']}")
    print(f"序列长度: {config['seq_length']}")
    
    # ========== 数据加载 ==========
    print("\n📊 加载增强子预测数据...")
    data_loaders, val_labels, test_labels = create_enhancer_data_loaders(
        train_seq_path=config['train_seq_path'],
        test_seq_path=config['test_seq_path'],
        batch_size=config['batch_size'],
        val_ratio=config['val_ratio'],
        random_state=config['random_state']
    )
    
    # 调试数据形状
    debug_data_shapes(data_loaders)
    
    # ========== 模型创建 ==========
    print("\n🧠 创建增强子预测模型...")
    model = create(
        channel1=config['channel1'],
        channel2=config['channel2'],
        channel3=config['channel3'],
        channel4=config['channel4'],
        channel5=config['channel5'],
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # ========== 开始训练 ==========
    print("\n🏋️ 开始增强子预测训练...")
    CREATE_enhancer_train(
        clf=model,
        train_loader=data_loaders['train'],
        valid_loader=data_loaders['val'],
        test_loader=data_loaders['test'],
        valid_labels=val_labels,
        test_labels=test_labels,
        lr=config['lr'],
        max_epoch=config['max_epoch'],
        pre_epoch=config['pre_epoch'],
        seq_loss_weight=config['seq_loss_weight'],
        outdir=config['outdir'],
        device=config['device']
    )


if __name__ == "__main__":
    main()