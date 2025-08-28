#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import copy
from sklearn import metrics

from utils import *

def calculate_forgetting_rate(original_scores, new_scores, original_labels):
    """计算遗忘率"""
    original_auc = metrics.roc_auc_score(original_labels, original_scores)
    new_auc = metrics.roc_auc_score(original_labels, new_scores)
    forgetting_rate = max(0, (original_auc - new_auc) / original_auc)
    return forgetting_rate, original_auc, new_auc

def elastic_weight_consolidation_loss(model, original_params, fisher_info, lambda_ewc=5000):
    """弹性权重巩固(EWC)损失"""
    ewc_loss = 0
    for name, param in model.named_parameters():
        if name in original_params:
            fisher = fisher_info.get(name, 0)
            ewc_loss += (fisher * (param - original_params[name]).pow(2)).sum()
    return lambda_ewc * ewc_loss

def compute_fisher_information(model, dataloader, device):
    """计算Fisher信息矩阵"""
    fisher_info = {}
    model.eval()
    
    for name, param in model.named_parameters():
        fisher_info[name] = torch.zeros_like(param)
    
    num_samples = 0
    for seq1, seq2, labels in dataloader:
        seq1, seq2, labels = seq1.to(device), seq2.to(device), labels.squeeze().to(device)
        
        model.zero_grad()
        outputs = model(seq1, seq2, task='all')
        
        # 计算所有任务的总损失
        clf_loss_fn = nn.BCEWithLogitsLoss()
        total_loss = (clf_loss_fn(outputs['epi_score'].squeeze(), labels.float()) + 
                     clf_loss_fn(outputs['eei_score'].squeeze(), labels.float()))
        
        total_loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_info[name] += param.grad.pow(2)
        
        num_samples += seq1.size(0)
    
    # 归一化Fisher信息
    for name in fisher_info:
        fisher_info[name] /= num_samples
    
    return fisher_info

def continual_learning_train(
    model,
    ctcf_loaders,
    epi_test_loader,
    eei_test_loader,
    epi_test_labels,
    eei_test_labels,
    sampling_ratio,
    lr=5e-6,
    max_epoch=100,
    device='cuda',
    outdir='./results/',
    use_ewc=True,
    lambda_ewc=5000
):
    """持续学习训练CTCF任务"""
    
    print(f"\n{'='*60}")
    print(f"开始持续学习训练 - 采样率: {sampling_ratio*100}%")
    print(f"{'='*60}")
    
    if sampling_ratio == 0.0:
        print("采样率为0%，直接使用预训练模型进行预测...")
        
        # 直接使用预训练模型预测
        model.eval()
        ctcf_scores = []
        with torch.no_grad():
            for seq1, seq2, _ in ctcf_loaders[sampling_ratio]['test']:
                seq1, seq2 = seq1.to(device), seq2.to(device)
                outputs = model(seq1, seq2, task='ctcf')
                ctcf_scores.extend(torch.sigmoid(outputs['ctcf_score'].squeeze()).cpu().numpy())
        
        ctcf_auc = metrics.roc_auc_score(ctcf_loaders[sampling_ratio]['test_labels'], ctcf_scores)
        
        # 计算原始任务性能
        epi_scores, eei_scores = evaluate_original_tasks(model, epi_test_loader, eei_test_loader, device)
        epi_auc = metrics.roc_auc_score(epi_test_labels, epi_scores)
        eei_auc = metrics.roc_auc_score(eei_test_labels, eei_scores)
        
        return {
            'ctcf_auc': ctcf_auc,
            'epi_auc': epi_auc,
            'eei_auc': eei_auc,
            'epi_forgetting': 0.0,
            'eei_forgetting': 0.0
        }
    
    # 记录原始模型参数（用于EWC）
    original_params = {}
    for name, param in model.named_parameters():
        original_params[name] = param.clone().detach()
    
    # 计算原始任务的性能
    print("计算原始任务性能...")
    original_epi_scores, original_eei_scores = evaluate_original_tasks(model, epi_test_loader, eei_test_loader, device)
    original_epi_auc = metrics.roc_auc_score(epi_test_labels, original_epi_scores)
    original_eei_auc = metrics.roc_auc_score(eei_test_labels, original_eei_scores)
    
    print(f"原始EPI AUC: {original_epi_auc:.4f}")
    print(f"原始EEI AUC: {original_eei_auc:.4f}")
    
    # 计算Fisher信息矩阵（如果使用EWC）
    fisher_info = {}
    if use_ewc and sampling_ratio > 0:
        print("计算Fisher信息矩阵...")
        # 使用EPI和EEI的验证数据计算Fisher信息
        # 这里需要从原始的训练函数中获取验证数据，暂时跳过Fisher信息计算
        pass
    
    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-7)
    
    clf_loss_fn = nn.BCEWithLogitsLoss()
    recon_loss_fn = nn.MSELoss()
    
    best_ctcf_auc = 0.0
    best_epoch = 0
    
    # 训练循环
    with tqdm(range(max_epoch), desc=f'CL Training (Sampling {sampling_ratio*100}%)') as pbar:
        for epoch in pbar:
            model.train()
            epoch_losses = []
            ctcf_preds = []
            ctcf_labels = []
            
            for seq1, seq2, labels in ctcf_loaders[sampling_ratio]['train']:
                seq1, seq2, labels = seq1.to(device), seq2.to(device), labels.squeeze().to(device)
                
                optimizer.zero_grad()
                
                # 前向传播
                outputs = model(seq1, seq2, task='ctcf')
                
                # 分类损失
                clf_loss = clf_loss_fn(outputs['ctcf_score'].squeeze(), labels.float())
                
                # 重构损失
                recon_loss = (recon_loss_fn(outputs['seq1_recon'], seq1) + 
                             recon_loss_fn(outputs['seq2_recon'], seq2))
                
                # EWC损失
                ewc_loss = 0
                if use_ewc and fisher_info:
                    ewc_loss = elastic_weight_consolidation_loss(model, original_params, fisher_info, lambda_ewc)
                
                # 总损失（增加EWC权重以减少遗忘）
                total_loss = clf_loss + 0.1 * recon_loss + ewc_loss
                
                # 反向传播
                total_loss.backward()
                optimizer.step()
                
                epoch_losses.append(total_loss.item())
                ctcf_preds.extend(torch.sigmoid(outputs['ctcf_score'].squeeze()).cpu().detach().numpy())
                ctcf_labels.extend(labels.cpu().numpy())
            
            # 验证
            if ctcf_loaders[sampling_ratio]['val'] is not None:
                model.eval()
                val_ctcf_scores = []
                val_losses = []
                
                with torch.no_grad():
                    for seq1, seq2, labels in ctcf_loaders[sampling_ratio]['val']:
                        seq1, seq2, labels = seq1.to(device), seq2.to(device), labels.squeeze().to(device)
                        outputs = model(seq1, seq2, task='ctcf')
                        
                        val_loss = clf_loss_fn(outputs['ctcf_score'].squeeze(), labels.float())
                        val_losses.append(val_loss.item())
                        val_ctcf_scores.extend(torch.sigmoid(outputs['ctcf_score'].squeeze()).cpu().numpy())
                
                val_ctcf_auc = metrics.roc_auc_score(ctcf_loaders[sampling_ratio]['val_labels'], val_ctcf_scores)
                
                if val_ctcf_auc > best_ctcf_auc:
                    best_ctcf_auc = val_ctcf_auc
                    best_epoch = epoch
                    
                    # 保存最佳模型
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch': epoch,
                        'ctcf_auc': val_ctcf_auc,
                        'sampling_ratio': sampling_ratio
                    }, os.path.join(outdir, f'best_ctcf_model_{sampling_ratio}.pth'))
                
                avg_loss = np.mean(epoch_losses)
                avg_val_loss = np.mean(val_losses)
                scheduler.step(avg_val_loss)
                
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Val_AUC': f'{val_ctcf_auc:.4f}',
                    'Best': f'{best_ctcf_auc:.4f}'
                })
                
                # 早停
                if epoch - best_epoch > 20:
                    print(f"早停于epoch {epoch}")
                    break
    
    # 加载最佳模型
    if os.path.exists(os.path.join(outdir, f'best_ctcf_model_{sampling_ratio}.pth')):
        checkpoint = torch.load(os.path.join(outdir, f'best_ctcf_model_{sampling_ratio}.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # 最终测试
    model.eval()
    test_ctcf_scores = []
    with torch.no_grad():
        for seq1, seq2, _ in ctcf_loaders[sampling_ratio]['test']:
            seq1, seq2 = seq1.to(device), seq2.to(device)
            outputs = model(seq1, seq2, task='ctcf')
            test_ctcf_scores.extend(torch.sigmoid(outputs['ctcf_score'].squeeze()).cpu().numpy())
    
    ctcf_auc = metrics.roc_auc_score(ctcf_loaders[sampling_ratio]['test_labels'], test_ctcf_scores)
    
    # 计算遗忘率
    print("计算原始任务的遗忘率...")
    new_epi_scores, new_eei_scores = evaluate_original_tasks(model, epi_test_loader, eei_test_loader, device)
    
    epi_forgetting, _, new_epi_auc = calculate_forgetting_rate(original_epi_scores, new_epi_scores, epi_test_labels)
    eei_forgetting, _, new_eei_auc = calculate_forgetting_rate(original_eei_scores, new_eei_scores, eei_test_labels)
    
    print(f"CTCF测试AUC: {ctcf_auc:.4f}")
    print(f"EPI AUC: {original_epi_auc:.4f} -> {new_epi_auc:.4f} (遗忘率: {epi_forgetting:.4f})")
    print(f"EEI AUC: {original_eei_auc:.4f} -> {new_eei_auc:.4f} (遗忘率: {eei_forgetting:.4f})")
    
    return {
        'ctcf_auc': ctcf_auc,
        'epi_auc': new_epi_auc,
        'eei_auc': new_eei_auc,
        'epi_forgetting': epi_forgetting,
        'eei_forgetting': eei_forgetting,
        'original_epi_auc': original_epi_auc,
        'original_eei_auc': original_eei_auc
    }

def evaluate_original_tasks(model, epi_test_loader, eei_test_loader, device):
    """评估原始任务性能"""
    model.eval()
    epi_scores = []
    eei_scores = []
    
    with torch.no_grad():
        # 评估EPI任务
        for seq1, seq2, _ in epi_test_loader:
            seq1, seq2 = seq1.to(device), seq2.to(device)
            outputs = model(seq1, seq2, task='epi')
            epi_scores.extend(torch.sigmoid(outputs['epi_score'].squeeze()).cpu().numpy())
        
        # 评估EEI任务
        for seq1, seq2, _ in eei_test_loader:
            seq1, seq2 = seq1.to(device), seq2.to(device)
            outputs = model(seq1, seq2, task='eei')
            eei_scores.extend(torch.sigmoid(outputs['eei_score'].squeeze()).cpu().numpy())
    
    return np.array(epi_scores), np.array(eei_scores)

def train_single_task_ctcf(
    ctcf_loaders,
    sampling_ratio,
    model_config,
    lr=1e-4,
    max_epoch=100,
    device='cuda',
    outdir='./results/'
):
    """训练单任务CTCF模型（用于对比）"""
    from layer import create
    
    print(f"\n{'='*60}")
    print(f"训练单任务CTCF模型 - 采样率: {sampling_ratio*100}%")
    print(f"{'='*60}")
    
    if sampling_ratio == 0.0:
        print("采样率为0%，无法训练单任务模型")
        return {'ctcf_auc': 0.0}
    
    # 创建新的单任务模型
    model = create(**model_config).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-7)
    
    clf_loss_fn = nn.BCEWithLogitsLoss()
    recon_loss_fn = nn.MSELoss()
    regloss = Regularization(model, weight_decay1=1e-8, weight_decay2=5e-7)
    
    best_ctcf_auc = 0.0
    best_epoch = 0
    
    with tqdm(range(max_epoch), desc=f'Single Task Training (Sampling {sampling_ratio*100}%)') as pbar:
        for epoch in pbar:
            model.train()
            epoch_losses = []
            
            for seq1, seq2, labels in ctcf_loaders[sampling_ratio]['train']:
                seq1, seq2, labels = seq1.to(device), seq2.to(device), labels.squeeze().to(device)
                
                optimizer.zero_grad()
                
                outputs = model(seq1, seq2, task='ctcf')
                
                # 分类损失
                clf_loss = clf_loss_fn(outputs['ctcf_score'].squeeze(), labels.float())
                
                # 重构损失
                recon_loss = (recon_loss_fn(outputs['seq1_recon'], seq1) + 
                             recon_loss_fn(outputs['seq2_recon'], seq2))
                
                # 正则化损失
                reg_loss = regloss(model)
                
                # 总损失
                total_loss = clf_loss + 0.1 * recon_loss + reg_loss
                
                total_loss.backward()
                optimizer.step()
                
                epoch_losses.append(total_loss.item())
            
            # 验证
            if ctcf_loaders[sampling_ratio]['val'] is not None:
                model.eval()
                val_scores = []
                val_losses = []
                
                with torch.no_grad():
                    for seq1, seq2, labels in ctcf_loaders[sampling_ratio]['val']:
                        seq1, seq2, labels = seq1.to(device), seq2.to(device), labels.squeeze().to(device)
                        outputs = model(seq1, seq2, task='ctcf')
                        
                        val_clf_loss = clf_loss_fn(outputs['ctcf_score'].squeeze(), labels.float())
                        val_recon_loss = (recon_loss_fn(outputs['seq1_recon'], seq1) + 
                                         recon_loss_fn(outputs['seq2_recon'], seq2))
                        val_reg_loss = regloss(model)
                        val_total_loss = val_clf_loss + 0.1 * val_recon_loss + val_reg_loss
                        
                        val_losses.append(val_total_loss.item())
                        val_scores.extend(torch.sigmoid(outputs['ctcf_score'].squeeze()).cpu().numpy())
                
                val_auc = metrics.roc_auc_score(ctcf_loaders[sampling_ratio]['val_labels'], val_scores)
                
                if val_auc > best_ctcf_auc:
                    best_ctcf_auc = val_auc
                    best_epoch = epoch
                    
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch': epoch,
                        'ctcf_auc': val_auc,
                        'sampling_ratio': sampling_ratio
                    }, os.path.join(outdir, f'best_single_ctcf_model_{sampling_ratio}.pth'))
                
                avg_loss = np.mean(epoch_losses)
                avg_val_loss = np.mean(val_losses)
                scheduler.step(avg_val_loss)
                
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Val_AUC': f'{val_auc:.4f}',
                    'Best': f'{best_ctcf_auc:.4f}'
                })
                
                if epoch - best_epoch > 20:
                    break
    
    # 加载最佳模型并测试
    if os.path.exists(os.path.join(outdir, f'best_single_ctcf_model_{sampling_ratio}.pth')):
        checkpoint = torch.load(os.path.join(outdir, f'best_single_ctcf_model_{sampling_ratio}.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    test_scores = []
    with torch.no_grad():
        for seq1, seq2, _ in ctcf_loaders[sampling_ratio]['test']:
            seq1, seq2 = seq1.to(device), seq2.to(device)
            outputs = model(seq1, seq2, task='ctcf')
            test_scores.extend(torch.sigmoid(outputs['ctcf_score'].squeeze()).cpu().numpy())
    
    ctcf_auc = metrics.roc_auc_score(ctcf_loaders[sampling_ratio]['test_labels'], test_scores)
    
    print(f"单任务CTCF测试AUC: {ctcf_auc:.4f}")
    
    return {'ctcf_auc': ctcf_auc}