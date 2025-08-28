#!/usr/bin/env python
import os
import torch
import numpy as np
import pandas as pd
import json
from data_loader import UniLoopNet_multitask_data_loaders, create_ctcf_data_loaders
from layer import create
from UniLoopNet_train import CREATE_multitask_train
from continual_learning import continual_learning_train, train_single_task_ctcf


def main():
    # ========== 配置参数 ==========
    config = {
        # 原始任务数据路径 - EPI任务
        'epi_train_seq1_path': '/public/home/shenyin_wsb_2606/Third/EP1000/5000/EPL_B.npz',
        'epi_train_seq2_path': '/public/home/shenyin_wsb_2606/Third/EP1000/5000/EPR_B.npz', 
        'epi_test_seq1_path': '/public/home/shenyin_wsb_2606/Third/EP1000/5000/EPL_C.npz',
        'epi_test_seq2_path': '/public/home/shenyin_wsb_2606/Third/EP1000/5000/EPR_C.npz',
        
        # 原始任务数据路径 - EEI任务
        'eei_train_seq1_path': '/public/home/shenyin_wsb_2606/Third/EE1000/5000/EEL_B.npz',
        'eei_train_seq2_path': '/public/home/shenyin_wsb_2606/Third/EE1000/5000/EER_B.npz',
        'eei_test_seq1_path': '/public/home/shenyin_wsb_2606/Third/EE1000/5000/EEL_C.npz',
        'eei_test_seq2_path': '/public/home/shenyin_wsb_2606/Third/EE1000/5000/EER_C.npz',
        
        
        'ctcf_train_seq1_path': '/public/home/shenyin_wsb_2606/Third/CTCF/CTCFL_B.npz',
        'ctcf_train_seq2_path': '/public/home/shenyin_wsb_2606/Third/CTCF/CTCFR_B.npz',
        'ctcf_test_seq1_path': '/public/home/shenyin_wsb_2606/Third/CTCF/CTCFL_C.npz',
        'ctcf_test_seq2_path': '/public/home/shenyin_wsb_2606/Third/CTCF/CTCFR_C.npz',
        
        # # 新任务数据路径 - CTCF任务（需要根据实际路径修改）
        # 'ctcf_train_seq1_path': '/public/home/shenyin_wsb_2606/Third/ctcf/1000/ctcfL_B.npz',
        # 'ctcf_train_seq2_path': '/public/home/shenyin_wsb_2606/Third/ctcf/1000/ctcfR_B.npz',
        # 'ctcf_test_seq1_path': '/public/home/shenyin_wsb_2606/Third/ctcf/1000/ctcfL_C.npz',
        # 'ctcf_test_seq2_path': '/public/home/shenyin_wsb_2606/Third/ctcf/1000/ctcfR_C.npz',
        
        # 训练参数
        'batch_size': 24,
        'val_ratio': 0.1,
        'lr': 3e-5,
        'max_epoch': 300,
        'pre_epoch': 50,
        'seq_loss_weight': 1.0,
        
        # 多任务权重
        'task_weights': {'epi': 1.0, 'eei': 1.0},
        
        # 持续学习参数
        'cl_lr': 1e-5,
        'cl_max_epoch': 100,
        'sampling_ratios': [0.0, 0.02,0.2,1.0],
        'use_ewc': True,
        'lambda_ewc': 1000,
        
        # CTCF数据填充参数
        'ctcf_target_length': 5000,  # 目标长度，与EPI/EEI保持一致
        'ctcf_original_length': 1000,  # CTCF原始长度
        'padding_strategy': 'symmetric',  # 填充策略：'symmetric', 'zero', 'edge'
        
        # 模型参数
        'channel1': 512,
        'channel2': 384,
        'channel3': 128,
        'channel4': 200,
        'channel5': 200,
        'embed_dim': 128,
        'seq_length': 5000,
        
        # 其他
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'outdir': './continual_learning_results/',
        'random_state': 42
    }
    
    print("🚀 开始EPI+EEI双任务预训练 + CTCF持续学习实验")
    print(f"使用设备: {config['device']}")
    print(f"输出目录: {config['outdir']}")
    
    # 创建输出目录
    os.makedirs(config['outdir'], exist_ok=True)
    os.makedirs(os.path.join(config['outdir'], 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(config['outdir'], 'results'), exist_ok=True)
    
    # ========== 步骤1: 预训练EPI+EEI双任务模型 ==========
    print(f"\n{'='*80}")
    print("步骤1: 预训练EPI+EEI双任务模型")
    print(f"{'='*80}")
    
    # 加载原始任务数据
    print("加载EPI+EEI双任务数据...")
    task_loaders, task_val_labels, task_test_labels = create_multitask_data_loaders(
        epi_train_seq1_path=config['epi_train_seq1_path'],
        epi_train_seq2_path=config['epi_train_seq2_path'],
        epi_test_seq1_path=config['epi_test_seq1_path'],
        epi_test_seq2_path=config['epi_test_seq2_path'],
        
        eei_train_seq1_path=config['eei_train_seq1_path'],
        eei_train_seq2_path=config['eei_train_seq2_path'],
        eei_test_seq1_path=config['eei_test_seq1_path'],
        eei_test_seq2_path=config['eei_test_seq2_path'],
        
        batch_size=config['batch_size'],
        val_ratio=config['val_ratio'],
        random_state=config['random_state']
    )
    
    # 创建模型
    print("创建EPI+EEI+CTCF多任务模型...")
    model = create(
        channel1=config['channel1'],
        channel2=config['channel2'],
        channel3=config['channel3'],
        channel4=config['channel4'],
        channel5=config['channel5'],
        embed_dim=config['embed_dim']
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 预训练EPI+EEI任务
    print("开始预训练EPI+EEI双任务...")
    pretrain_outdir = os.path.join(config['outdir'], 'pretrain')
    os.makedirs(pretrain_outdir, exist_ok=True)
    
    # CREATE_multitask_train(
    #     clf=model,
    #     train_loader=task_loaders,
    #     valid_loader=task_loaders,
    #     test_loader=task_loaders,
    #     valid_labels=task_val_labels,
    #     test_labels=task_test_labels,
    #     lr=config['lr'],
    #     max_epoch=config['max_epoch'],
    #     pre_epoch=config['pre_epoch'],
    #     seq_loss_weight=config['seq_loss_weight'],
    #     task_weights=config['task_weights'],
    #     outdir=pretrain_outdir,
    #     device=config['device']
    # )
    
    # 加载最佳预训练模型
    pretrain_model_path = os.path.join(pretrain_outdir, 'checkpoint', 'best_model.pth')
    if os.path.exists(pretrain_model_path):
        print("加载最佳预训练模型...")
        checkpoint = torch.load(pretrain_model_path, map_location=config['device'])
        model.load_state_dict(checkpoint['model'])
        print(f"加载预训练模型成功 (Epoch {checkpoint['epoch']}, AUC: {checkpoint['val_avg_auc']:.4f})")
    else:
        print("警告: 未找到预训练模型，使用当前模型继续...")
    
    # ========== 步骤2: 加载CTCF数据 ==========
    print(f"\n{'='*80}")
    print("步骤2: 加载CTCF任务数据")
    print(f"{'='*80}")
    
    # 加载CTCF数据
    print("加载CTCF任务数据...")
    ctcf_loaders = create_ctcf_data_loaders(
        ctcf_train_seq1_path=config['ctcf_train_seq1_path'],
        ctcf_train_seq2_path=config['ctcf_train_seq2_path'],
        ctcf_test_seq1_path=config['ctcf_test_seq1_path'],
        ctcf_test_seq2_path=config['ctcf_test_seq2_path'],
        sampling_ratios=config['sampling_ratios'],
        batch_size=config['batch_size'],
        val_ratio=config['val_ratio'],
        random_state=config['random_state'],
        target_length=config['ctcf_target_length'],
        original_length=config['ctcf_original_length'],
        padding_strategy=config['padding_strategy']
    )
    
    # ========== 步骤3: 持续学习实验 ==========
    print(f"\n{'='*80}")
    print("步骤3: 持续学习实验")
    print(f"{'='*80}")
    
    # 保存结果
    cl_results = []
    single_task_results = []
    
    # 模型配置（用于单任务训练）
    model_config = {
        'channel1': config['channel1'],
        'channel2': config['channel2'],
        'channel3': config['channel3'],
        'channel4': config['channel4'],
        'channel5': config['channel5'],
        'embed_dim': config['embed_dim']
    }
    
    for sampling_ratio in config['sampling_ratios']:
        print(f"\n{'-'*60}")
        print(f"实验采样率: {sampling_ratio*100}%")
        print(f"{'-'*60}")
        
        # 为每个采样率创建模型副本
        model_copy = create(**model_config)
        model_copy.load_state_dict(model.state_dict())
        model_copy = model_copy.to(config['device'])
        
        # 持续学习训练
        cl_result = continual_learning_train(
            model=model_copy,
            ctcf_loaders=ctcf_loaders,
            epi_test_loader=task_loaders['epi']['test'],
            eei_test_loader=task_loaders['eei']['test'],
            epi_test_labels=task_test_labels['epi'],
            eei_test_labels=task_test_labels['eei'],
            sampling_ratio=sampling_ratio,
            lr=config['cl_lr'],
            max_epoch=config['cl_max_epoch'],
            device=config['device'],
            outdir=config['outdir'],
            use_ewc=config['use_ewc'],
            lambda_ewc=config['lambda_ewc']
        )
        
        cl_result['sampling_ratio'] = sampling_ratio
        cl_result['method'] = 'continual_learning'
        cl_results.append(cl_result)
        
        # 单任务训练（用于对比）
        if sampling_ratio > 0:  # 0%采样率无法训练单任务模型
            single_task_result = train_single_task_ctcf(
                ctcf_loaders=ctcf_loaders,
                sampling_ratio=sampling_ratio,
                model_config=model_config,
                lr=config['cl_lr'] * 10,  # 单任务使用更高学习率
                max_epoch=config['cl_max_epoch'],
                device=config['device'],
                outdir=config['outdir']
            )
            
            single_task_result['sampling_ratio'] = sampling_ratio
            single_task_result['method'] = 'single_task'
            single_task_results.append(single_task_result)
    
    # ========== 步骤4: 结果分析 ==========
    print(f"\n{'='*80}")
    print("步骤4: 实验结果分析")
    print(f"{'='*80}")
    
    # 转换为DataFrame
    cl_df = pd.DataFrame(cl_results)
    single_task_df = pd.DataFrame(single_task_results)
    
    # 打印结果表格
    print("\n持续学习结果:")
    print("="*120)
    print(f"{'采样率':<8} {'CTCF AUC':<12} {'EPI AUC':<10} {'EEI AUC':<10} {'EPI遗忘率':<12} {'EEI遗忘率':<12}")
    print("="*120)
    
    for _, row in cl_df.iterrows():
        print(f"{row['sampling_ratio']*100:>6.0f}%  "
              f"{row['ctcf_auc']:>10.4f}  "
              f"{row['epi_auc']:>8.4f}  "
              f"{row['eei_auc']:>8.4f}  "
              f"{row['epi_forgetting']:>10.4f}  "
              f"{row['eei_forgetting']:>10.4f}")
    
    print("\n单任务训练结果:")
    print("="*40)
    print(f"{'采样率':<8} {'CTCF AUC':<12}")
    print("="*40)
    
    for _, row in single_task_df.iterrows():
        print(f"{row['sampling_ratio']*100:>6.0f}%  {row['ctcf_auc']:>10.4f}")
    
    # 对比分析
    print("\n持续学习 vs 单任务训练对比:")
    print("="*60)
    print(f"{'采样率':<8} {'持续学习':<12} {'单任务':<10} {'提升':<10}")
    print("="*60)
    
    for cl_row in cl_results:
        if cl_row['sampling_ratio'] > 0:
            single_row = next((x for x in single_task_results if x['sampling_ratio'] == cl_row['sampling_ratio']), None)
            if single_row:
                improvement = cl_row['ctcf_auc'] - single_row['ctcf_auc']
                print(f"{cl_row['sampling_ratio']*100:>6.0f}%  "
                      f"{cl_row['ctcf_auc']:>10.4f}  "
                      f"{single_row['ctcf_auc']:>8.4f}  "
                      f"{improvement:>+8.4f}")
    
    # 保存结果
    results_file = os.path.join(config['outdir'], 'results', 'experiment_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'config': config,
            'continual_learning_results': cl_results,
            'single_task_results': single_task_results,
        }, f, indent=2)
    
    # 保存CSV
    cl_df.to_csv(os.path.join(config['outdir'], 'results', 'continual_learning_results.csv'), index=False)
    if not single_task_df.empty:
        single_task_df.to_csv(os.path.join(config['outdir'], 'results', 'single_task_results.csv'), index=False)
    
    print(f"\n实验完成! 结果保存在: {config['outdir']}")
    print(f"详细结果文件: {results_file}")


if __name__ == "__main__":
    main()