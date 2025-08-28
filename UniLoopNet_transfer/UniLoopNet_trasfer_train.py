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

def CREATE_enhancer_train(
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
        outdir='./output/', 
        device='cuda'
    ):
    """
    增强子预测模型训练
    """
    # 创建输出目录
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, 'checkpoint'), exist_ok=True)
    
    clf = clf.to(device)
    
    # 转换标签格式
    valid_labels = np.array(valid_labels).astype(int)
    test_labels = np.array(test_labels).astype(int)
    
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
    max_auc = 0.0
    max_auc_epoch = 0
    
    # 训练记录
    train_history = {'loss': [], 'auc': []}
    val_history = {'loss': [], 'auc': []}
    
    with tqdm(range(max_epoch), total=max_epoch, desc='Epochs') as tq:
        for epoch in tq:
            # 训练输出收集
            train_out = []
            train_labels_list = []
            training_loss, training_clf, training_recon = [], [], []
            
            # ========== 训练阶段 ==========
            clf.train()
            
            for seq, labels in train_loader:
                # 收集标签
                train_labels_list.extend(labels.squeeze().cpu().numpy())
                
                # 移动到设备
                seq = seq.to(device)
                labels = labels.squeeze().to(device)
                
                # 模型前向传播
                outputs = clf(seq)
                
                # 收集预测结果
                train_out.extend(torch.sigmoid(outputs['enhancer_score'].squeeze()).cpu().detach().numpy())

                # 正则化损失
                reg_loss = regloss(clf)
                
                # 分类损失
                clf_loss = clf_loss_fn(outputs['enhancer_score'].squeeze(), labels.float()) + reg_loss
                
                # 重构损失
                recon_loss = seq_loss_weight * recon_loss_fn(outputs['seq_recon'], seq)
                
                # 总损失
                total_loss = loss1_rate[epoch] * clf_loss + loss2_rate[epoch] * recon_loss
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # 记录损失
                training_loss.append(total_loss.item())
                training_clf.append(clf_loss.item())
                training_recon.append(recon_loss.item())
                    
            # 计算训练指标
            train_loss = float(np.mean(training_loss))
            train_recon = float(np.mean(training_recon))
            train_clf = float(np.mean(training_clf))
            
            # 转换为numpy数组
            train_scores = np.array(train_out)
            train_labels_np = np.array(train_labels_list).astype(int)
            
            # ========== 验证阶段 ==========
            clf.eval()
            val_out = []
            val_loss_list = []
            
            with torch.no_grad():
                for seq, labels in valid_loader:
                    seq = seq.to(device)
                    labels = labels.squeeze().to(device)
                    
                    outputs = clf(seq)
                    
                    val_out.extend(torch.sigmoid(outputs['enhancer_score'].squeeze()).cpu().numpy())
                    
                    val_clf_loss = clf_loss_fn(outputs['enhancer_score'].squeeze(), labels.float())
                    val_recon_loss = recon_loss_fn(outputs['seq_recon'], seq)
                    val_total_loss = val_clf_loss + val_recon_loss
                    val_loss_list.append(val_total_loss.item())
                            
            val_scores = np.array(val_out)
            val_loss = np.mean(val_loss_list)
            
            # ========== 计算评估指标 ==========
            if epoch >= pre_epoch:
                # 计算AUC
                train_auc = metrics.roc_auc_score(train_labels_np, train_scores)
                val_auc = metrics.roc_auc_score(valid_labels, val_scores)
                
                print(f"\nEpoch {epoch+1}/{max_epoch}")
                print(f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
            else:
                val_auc = 0.0
            
            # 进度条信息
            if epoch < pre_epoch:
                epoch_info = f'recon={train_recon:.3f}'
            else:
                epoch_info = f'recon={train_recon:.3f}, clf={train_clf:.3f}, val_auc={val_auc:.3f}'
            tq.set_postfix_str(epoch_info)
    
            # 学习率调度
            if epoch < pre_epoch:
                scheduler1.step(train_recon)
            else:
                scheduler2.step(val_loss)
    
            # 模型保存和早停
            if epoch >= pre_epoch and val_auc > max_auc:
                max_auc = val_auc
                max_auc_epoch = epoch
                
                # 保存最佳模型
                state = {
                    'model': clf.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_auc': val_auc,
                    'train_history': train_history,
                    'val_history': val_history
                }
                
                torch.save(state, os.path.join(outdir, 'checkpoint', 'best_model.pth'))
                print(f"保存最佳模型 (Epoch {epoch+1}, AUC: {val_auc:.4f})")
            
            # 早停检查
            if epoch >= pre_epoch and epoch - max_auc_epoch > 30:
                print(f"早停: {30} epochs without improvement")
                break
    
    # ========== 测试阶段 ==========
    print("\n🚀 进行最终测试...")
    clf.eval()
    test_out = []
    with torch.no_grad():
        for seq, labels in test_loader:
            outputs = clf(seq.to(device))
            test_out.extend(torch.sigmoid(outputs['enhancer_score'].squeeze()).cpu().numpy())
                        
    test_scores = np.array(test_out)
    
    # 计算测试集指标
    test_auc = metrics.roc_auc_score(test_labels, test_scores)
    
    print(f"Test AUC: {test_auc:.4f}")
    
    print(f"\n训练完成!")
    print(f"最佳验证AUC: {max_auc:.4f} (Epoch {max_auc_epoch+1})")
    print(f"最终测试AUC: {test_auc:.4f}")
    
    return max_auc, test_auc