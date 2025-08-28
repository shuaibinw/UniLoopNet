
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from sklearn import metrics

from layer import *
from utils import *

def CREATE_multitask_train(
        clf, 
        train_loader, 
        valid_loader, 
        test_loader, 
        valid_labels, 
        test_labels, 
        lr=5e-5, 
        max_epoch=300, 
        pre_epoch=50, 
        seq_loss_weight=1.0,
        task_weights={'epi': 1.0, 'eei': 1.0},  # 只保留EPI和EEI
        outdir='./output/', 
        device='cuda'
    ):
    """
    EPI+EEI双任务增强子-启动子相互作用预测模型训练
    """
    # 创建输出目录
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, 'checkpoint'), exist_ok=True)
    
    clf = clf.to(device)
    
    # 转换标签格式
    valid_epi_labels = np.array(valid_labels['epi']).astype(int)
    valid_eei_labels = np.array(valid_labels['eei']).astype(int)
    
    test_epi_labels = np.array(test_labels['epi']).astype(int)
    test_eei_labels = np.array(test_labels['eei']).astype(int)
    
    optimizer = optim.Adam(clf.parameters(), lr=lr)
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, min_lr=1e-6)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20, min_lr=1e-8)
    
    # 损失函数
    clf_loss_fn = nn.BCEWithLogitsLoss()
    recon_loss_fn = nn.MSELoss()
    regloss = Regularization(clf, weight_decay1=1e-8, weight_decay2=5e-7)

    loss1_rate = [0.0] * pre_epoch + [1.0] * (max_epoch - pre_epoch)
    loss2_rate = [1.0] * pre_epoch + [0.5] * (max_epoch - pre_epoch)
    
    # 记录最佳性能
    max_avg_auc = 0.0
    max_auc_epoch = 0
    
    # 训练记录
    train_history = {'loss': [], 'epi_auc': [], 'eei_auc': []}
    val_history = {'loss': [], 'epi_auc': [], 'eei_auc': []}
    
    # 获取各任务的数据加载器 - 只保留EPI和EEI
    train_loaders = {
        'epi': train_loader['epi']['train'],
        'eei': train_loader['eei']['train']
    }
    
    valid_loaders = {
        'epi': valid_loader['epi']['val'],
        'eei': valid_loader['eei']['val']
    }
    
    test_loaders = {
        'epi': test_loader['epi']['test'],
        'eei': test_loader['eei']['test']
    }
    
    with tqdm(range(max_epoch), total=max_epoch, desc='Epochs') as tq:
        for epoch in tq:
            # 训练输出收集
            train_epi_out, train_eei_out = [], []
            train_epi_labels, train_eei_labels = [], []
            training_loss, training_clf, training_recon = [], [], []
            
            # ========== 训练阶段 ==========
            clf.train()
            
            # 创建双任务批次迭代器
            task_iters = {
                'epi': iter(train_loaders['epi']),
                'eei': iter(train_loaders['eei'])
            }
            
            # 计算最大批次数（取最长的数据加载器）
            max_batches = max(len(train_loaders['epi']), len(train_loaders['eei']))
            
            for batch_idx in range(max_batches):
                batch_losses = []
                batch_clf_losses = []
                batch_recon_losses = []
                
                # 处理每个任务 - 只保留EPI和EEI
                for task_name in ['epi', 'eei']:
                    try:
                        # 获取当前任务的批次数据
                        seq1, seq2, labels = next(task_iters[task_name])
                    except StopIteration:
                        # 如果当前任务的数据用完了，重新开始
                        task_iters[task_name] = iter(train_loaders[task_name])
                        seq1, seq2, labels = next(task_iters[task_name])
                    
                    # 收集标签
                    if task_name == 'epi':
                        train_epi_labels.extend(labels.squeeze().cpu().numpy())
                    elif task_name == 'eei':
                        train_eei_labels.extend(labels.squeeze().cpu().numpy())
                    
                    # 移动到设备
                    seq1 = seq1.to(device)
                    seq2 = seq2.to(device)
                    labels = labels.squeeze().to(device)
                    
                    # 模型前向传播
                    outputs = clf(seq1, seq2)
                    
                    # 收集预测结果
                    if task_name == 'epi':
                        train_epi_out.extend(torch.sigmoid(outputs['epi_score'].squeeze()).cpu().detach().numpy())
                    elif task_name == 'eei':
                        train_eei_out.extend(torch.sigmoid(outputs['eei_score'].squeeze()).cpu().detach().numpy())

                    # 正则化损失
                    reg_loss = regloss(clf)
                    
                    # 当前任务的分类损失
                    if task_name == 'epi':
                        task_clf_loss = clf_loss_fn(outputs['epi_score'].squeeze(), labels.float())
                    elif task_name == 'eei':
                        task_clf_loss = clf_loss_fn(outputs['eei_score'].squeeze(), labels.float())
                    
                    # 加权任务损失
                    weighted_clf_loss = task_weights[task_name] * task_clf_loss + reg_loss
                    
                    # 重构损失
                    seq1_recon_loss = seq_loss_weight * recon_loss_fn(outputs['seq1_recon'], seq1)
                    seq2_recon_loss = seq_loss_weight * recon_loss_fn(outputs['seq2_recon'], seq2)
                    recon_loss = seq1_recon_loss + seq2_recon_loss
                    
                    # 总损失
                    total_loss = loss1_rate[epoch] * weighted_clf_loss + loss2_rate[epoch] * recon_loss
                    
                    # 只有当任务权重大于0时才进行反向传播
                    if task_weights[task_name] > 0:
                        batch_losses.append(total_loss.item())
                        batch_clf_losses.append(weighted_clf_loss.item())
                        batch_recon_losses.append(recon_loss.item())
                        
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()
                
                # 记录批次统计
                if batch_losses:
                    training_loss.extend(batch_losses)
                    training_clf.extend(batch_clf_losses)
                    training_recon.extend(batch_recon_losses)
                    
            # 计算训练指标
            train_loss = float(np.mean(training_loss)) if training_loss else 0.0
            train_recon = float(np.mean(training_recon)) if training_recon else 0.0
            train_clf = float(np.mean(training_clf)) if training_clf else 0.0
            
            # 转换为numpy数组
            train_epi_scores = np.array(train_epi_out) if train_epi_out else np.array([])
            train_eei_scores = np.array(train_eei_out) if train_eei_out else np.array([])
            train_epi_labels_np = np.array(train_epi_labels).astype(int) if train_epi_labels else np.array([])
            train_eei_labels_np = np.array(train_eei_labels).astype(int) if train_eei_labels else np.array([])
            
            # ========== 验证阶段 ==========
            clf.eval()
            val_epi_out, val_eei_out = [], []
            val_loss_list = []
            
            with torch.no_grad():
                # 验证每个任务
                for task_name in ['epi', 'eei']:
                    if task_weights[task_name] > 0:  # 只验证有权重的任务
                        for seq1, seq2, labels in valid_loaders[task_name]:  # 修复：移除 Hawkins
                            seq1 = seq1.to(device)
                            seq2 = seq2.to(device)
                            labels = labels.squeeze().to(device)
                            
                            outputs = clf(seq1, seq2)
                            
                            if task_name == 'epi':
                                val_epi_out.extend(torch.sigmoid(outputs['epi_score'].squeeze()).cpu().numpy())
                                val_clf_loss = clf_loss_fn(outputs['epi_score'].squeeze(), labels.float())
                            elif task_name == 'eei':
                                val_eei_out.extend(torch.sigmoid(outputs['eei_score'].squeeze()).cpu().numpy())
                                val_clf_loss = clf_loss_fn(outputs['eei_score'].squeeze(), labels.float())
                            
                            val_recon_loss = (recon_loss_fn(outputs['seq1_recon'], seq1) + 
                                            recon_loss_fn(outputs['seq2_recon'], seq2))
                            val_total_loss = val_clf_loss + val_recon_loss
                            val_loss_list.append(val_total_loss.item())
                            
            val_epi_scores = np.array(val_epi_out) if val_epi_out else np.array([])
            val_eei_scores = np.array(val_eei_out) if val_eei_out else np.array([])
            val_loss = np.mean(val_loss_list) if val_loss_list else 0.0
            
            # ========== 计算评估指标 ==========
            if epoch >= pre_epoch:
                # 只计算有权重的任务的AUC
                train_aucs = []
                val_aucs = []
                
                if task_weights['epi'] > 0 and len(train_epi_scores) > 0:
                    train_epi_auc = metrics.roc_auc_score(train_epi_labels_np, train_epi_scores)
                    val_epi_auc = metrics.roc_auc_score(valid_epi_labels, val_epi_scores)
                    train_aucs.append(train_epi_auc)
                    val_aucs.append(val_epi_auc)
                    print(f"EPI - Train: {train_epi_auc:.4f}, Val: {val_epi_auc:.4f}")
                
                if task_weights['eei'] > 0 and len(train_eei_scores) > 0:
                    train_eei_auc = metrics.roc_auc_score(train_eei_labels_np, train_eei_scores)
                    val_eei_auc = metrics.roc_auc_score(valid_eei_labels, val_eei_scores)
                    train_aucs.append(train_eei_auc)
                    val_aucs.append(val_eei_auc)
                    print(f"EEI - Train: {train_eei_auc:.4f}, Val: {val_eei_auc:.4f}")
                
                # 计算平均AUC
                if train_aucs:
                    train_avg_auc = np.mean(train_aucs)
                    val_avg_auc = np.mean(val_aucs)
                    
                    print(f"\nEpoch {epoch+1}/{max_epoch}")
                    print(f"Average - Train: {train_avg_auc:.4f}, Val: {val_avg_auc:.4f}")
                else:
                    val_avg_auc = 0.0
            
            # 进度条信息
            if epoch < pre_epoch:
                epoch_info = f'recon={train_recon:.3f}'
            else:
                epoch_info = f'recon={train_recon:.3f}, clf={train_clf:.3f}, val_avg_auc={val_avg_auc:.3f}'
            tq.set_postfix_str(epoch_info)
    
            # 学习率调度
            if epoch < pre_epoch:
                scheduler1.step(train_recon)
            else:
                scheduler2.step(val_loss)
    
            # 模型保存和早停
            if epoch >= pre_epoch and val_avg_auc > max_avg_auc:
                max_avg_auc = val_avg_auc
                max_auc_epoch = epoch
                
                # 保存最佳模型
                state = {
                    'model': clf.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_avg_auc': val_avg_auc,
                    'train_history': train_history,
                    'val_history': val_history,
                    'task_weights': task_weights
                }
                
                torch.save(state, os.path.join(outdir, 'checkpoint', 'best_model.pth'))
                print(f"保存最佳模型 (Epoch {epoch+1}, Avg AUC: {val_avg_auc:.4f})")
            
            # 早停检查
            if epoch >= pre_epoch and epoch - max_auc_epoch > 30:
                print(f"早停: {30} epochs without improvement")
                break
    
    # ========== 测试阶段 ==========
    print("\n🚀 进行最终测试...")
    clf.eval()
    test_epi_out, test_eei_out = [], []
    with torch.no_grad():
        for task_name in ['epi', 'eei']:
            if task_weights[task_name] > 0:  # 只测试有权重的任务
                for seq1, seq2, labels in test_loaders[task_name]:
                    outputs = clf(seq1.to(device), seq2.to(device))
                    if task_name == 'epi':
                        test_epi_out.extend(torch.sigmoid(outputs['epi_score'].squeeze()).cpu().numpy())
                    elif task_name == 'eei':
                        test_eei_out.extend(torch.sigmoid(outputs['eei_score'].squeeze()).cpu().numpy())
                        
    test_epi_scores = np.array(test_epi_out) if test_epi_out else np.array([])
    test_eei_scores = np.array(test_eei_out) if test_eei_out else np.array([])
    
    # 计算测试集指标
    test_aucs = []
    if task_weights['epi'] > 0 and len(test_epi_scores) > 0:
        test_epi_auc = metrics.roc_auc_score(test_epi_labels, test_epi_scores)
        test_aucs.append(test_epi_auc)
        
        print(f"EPI - Test: {test_epi_auc:.4f}")
    
    if task_weights['eei'] > 0 and len(test_eei_scores) > 0:
        test_eei_auc = metrics.roc_auc_score(test_eei_labels, test_eei_scores)
        test_aucs.append(test_eei_auc)
        print(f"EEI - Test: {test_eei_auc:.4f}")
    
    if test_aucs:
        test_avg_auc = np.mean(test_aucs)
        print(f"Average - Test: {test_avg_auc:.4f}")
    
    print(f"\n训练完成!")
    print(f"最佳验证平均AUC: {max_avg_auc:.4f} (Epoch {max_auc_epoch+1})")
    print(f"最终测试平均AUC: {test_avg_auc:.4f}")
    
    return max_avg_auc, test_avg_auc