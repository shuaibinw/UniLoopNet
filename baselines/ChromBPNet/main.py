
#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import random as rn
import os

# 设置环境变量以确保可重复性
os.environ['PYTHONHASHSEED'] = '0'

# 设置随机种子
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    rn.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 数据集类
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
        seq1_raw = self.seq1_data[idx]  # shape: (20000,)
        seq2_raw = self.seq2_data[idx]  # shape: (20000,)
        seq1 = torch.FloatTensor(seq1_raw.reshape(5000, 4)).transpose(0, 1)  # (4, 5000)
        seq2 = torch.FloatTensor(seq2_raw.reshape(5000, 4)).transpose(0, 1)  # (4, 5000)
        label = torch.FloatTensor([self.labels[idx]])
        return seq1, seq2, label

# 数据加载函数
def load_task_data(seq1_path, seq2_path, task_name):
    """加载单个任务的数据"""
    print(f"加载{task_name.upper()}任务数据: {seq1_path}, {seq2_path}")
    
    seq1_data = np.load(seq1_path)
    seq1_seqs = seq1_data['sequence']  # shape: (n_samples, 20000)
    seq1_labels = seq1_data['label']   # shape: (n_samples,)
    
    seq2_data = np.load(seq2_path)
    seq2_seqs = seq2_data['sequence']  # shape: (n_samples, 20000)
    seq2_labels = seq2_data['label']   # shape: (n_samples,)
    
    assert len(seq1_seqs) == len(seq2_seqs), f"{task_name}任务的两个序列文件样本数量不一致"
    
    if not np.array_equal(seq1_labels, seq2_labels):
        print(f"警告: {task_name}任务的两个序列文件标签不完全一致，使用第一个文件的标签")
    
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

# 多任务数据加载器
def create_multitask_data_loaders(
    epi_train_seq1_path, epi_train_seq2_path,
    epi_test_seq1_path, epi_test_seq2_path,
    eei_train_seq1_path, eei_train_seq2_path,
    eei_test_seq1_path, eei_test_seq2_path,
    batch_size=32, val_ratio=0.1, random_state=42
):
    """创建EPI+EEI双任务数据加载器"""
    
    task_loaders = {}
    task_val_labels = {}
    task_test_labels = {}
    
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
        
        print(f"加载{task_name.upper()}训练数据...")
        train_seq1, train_seq2, train_labels = load_task_data(
            paths['train_seq1'], paths['train_seq2'], task_name
        )
        
        print(f"加载{task_name.upper()}测试数据...")
        test_seq1, test_seq2, test_labels = load_task_data(
            paths['test_seq1'], paths['test_seq2'], task_name
        )
        
        print(f"按 {1-val_ratio}:{val_ratio} 划分训练集和验证集...")
        indices = np.arange(len(train_labels))
        train_idx, val_idx = train_test_split(
            indices, 
            test_size=val_ratio, 
            random_state=random_state,
            stratify=train_labels
        )
        
        train_s1 = train_seq1[train_idx]
        train_s2 = train_seq2[train_idx]
        train_lab = train_labels[train_idx]
        val_s1 = train_seq1[val_idx]
        val_s2 = train_seq2[val_idx]
        val_lab = train_labels[val_idx]
        
        print(f"数据划分完成:")
        print(f"  - 训练集大小: {len(train_lab)} (正样本: {np.sum(train_lab)})")
        print(f"  - 验证集大小: {len(val_lab)} (正样本: {np.sum(val_lab)})")
        print(f"  - 测试集大小: {len(test_labels)} (正样本: {np.sum(test_labels)})")
        
        train_dataset = SingleTaskDataset(train_s1, train_s2, train_lab, task_name)
        val_dataset = SingleTaskDataset(val_s1, val_s2, val_lab, task_name)
        test_dataset = SingleTaskDataset(test_seq1, test_seq2, test_labels, task_name)
        
        task_loaders[task_name] = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
            'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        }
        
        task_val_labels[task_name] = val_lab
        task_test_labels[task_name] = test_labels
    
    return task_loaders, task_val_labels, task_test_labels

# BPNet模型
class BPNet(nn.Module):
    def __init__(self, model_params, args):
        super(BPNet, self).__init__()
        self.filters = int(model_params['filters'])
        self.n_dil_layers = int(model_params['n_dil_layers'])
        self.conv1_kernel_size = 21
        self.profile_kernel_size = 75
        self.sequence_len = int(model_params['inputlen'])
        self.out_pred_len = int(model_params['outputlen'])
        self.counts_loss_weight = float(model_params['counts_loss_weight'])
        
        set_seed(args.seed)
        
        # 第一卷积层
        self.conv1 = nn.Conv1d(4, self.filters, kernel_size=self.conv1_kernel_size, padding=0)
        self.relu = nn.ReLU()
        
        # 膨胀卷积层
        self.dilated_convs = nn.ModuleList()
        for i in range(1, self.n_dil_layers + 1):
            conv = nn.Conv1d(self.filters, self.filters, kernel_size=3, padding=0, dilation=2**i)
            self.dilated_convs.append(conv)
        
        # 轮廓预测分支
        self.prof_conv = nn.Conv1d(self.filters, 1, kernel_size=self.profile_kernel_size, padding=0)
        
        # 计数预测分支
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.count_dense = nn.Linear(self.filters, 1)
        
        # 二分类输出
        self.classifier = nn.Linear(self.out_pred_len + 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, seq1, seq2):
        x = (seq1 + seq2) / 2  # shape: (batch_size, 4, 5000)
        x = self.conv1(x)
        x = self.relu(x)
        
        for i in range(self.n_dil_layers):
            conv_x = self.dilated_convs[i](x)
            x_len = x.size(2)
            conv_x_len = conv_x.size(2)
            crop_size = (x_len - conv_x_len) // 2
            x = x[:, :, crop_size:crop_size + conv_x_len]  # 对称裁剪
            x = conv_x + x  # 残差连接
        
        prof_out = self.prof_conv(x)
        cropsize = (prof_out.size(2) - self.out_pred_len) // 2
        prof_out = prof_out[:, :, cropsize:cropsize + self.out_pred_len]
        prof_out = prof_out.view(prof_out.size(0), -1)  # (batch_size, out_pred_len)
        
        count_out = self.global_avg_pool(x)
        count_out = count_out.view(count_out.size(0), -1)
        count_out = self.count_dense(count_out)  # (batch_size, 1)
        
        combined = torch.cat([prof_out, count_out], dim=1)  # (batch_size, out_pred_len+1)
        class_out = self.classifier(combined)
        class_out = self.sigmoid(class_out)  # (batch_size, 1)
        
        return prof_out, count_out, class_out

# 损失函数
def combined_loss(prof_out, count_out, class_out, labels, counts_loss_weight):
    class_loss = nn.BCELoss()(class_out, labels)
    prof_loss = nn.MSELoss()(prof_out, torch.zeros_like(prof_out))
    count_loss = nn.MSELoss()(count_out, torch.zeros_like(count_out))
    return class_loss + counts_loss_weight * (prof_loss + count_loss)

# 绘制ROC和PRC曲线
def plot_roc_prc(task_name, labels, preds, split_name, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 确保preds和labels是NumPy数组
    preds = np.array(preds)
    labels = np.array(labels)
    
    # ROC曲线
    fpr, tpr, _ = roc_curve(labels, preds)
    auc_score = roc_auc_score(labels, preds)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{task_name.upper()} {split_name} ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{task_name}_{split_name}_roc.png'))
    plt.close()
    
    # PRC曲线
    precision, recall, _ = precision_recall_curve(labels, preds)
    prc_score = average_precision_score(labels, preds)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PRC (AP = {prc_score:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{task_name.upper()} {split_name} PRC Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{task_name}_{split_name}_prc.png'))
    plt.close()

# 计算评估指标
def compute_metrics(labels, preds, threshold=0.5):
    # 确保输入是NumPy数组
    preds = np.array(preds)
    labels = np.array(labels)
    
    binary_preds = (preds >= threshold).astype(int)
    auc = roc_auc_score(labels, preds)
    prc = average_precision_score(labels, preds)
    acc = accuracy_score(labels, binary_preds)
    f1 = f1_score(labels, binary_preds)
    precision = precision_score(labels, binary_preds)
    recall = recall_score(labels, binary_preds)
    return {'AUC': auc, 'PRC': prc, 'Accuracy': acc, 'F1': f1, 'Precision': precision, 'Recall': recall}

# 训练和评估函数
def train_and_evaluate(task_loaders, task_val_labels, task_test_labels, model_params, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for task_name, loaders in task_loaders.items():
        print(f"\n{'='*50}")
        print(f"训练 {task_name.upper()} 任务")
        print(f"{'='*50}")
        
        model = BPNet(model_params, args).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        num_epochs = 10
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            train_preds, train_labels = [], []
            
            for seq1, seq2, labels in loaders['train']:
                seq1, seq2, labels = seq1.to(device), seq2.to(device), labels.to(device)
                optimizer.zero_grad()
                prof_out, count_out, class_out = model(seq1, seq2)
                loss = combined_loss(prof_out, count_out, class_out, labels, model_params['counts_loss_weight'])
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                train_preds.extend(class_out.cpu().detach().numpy().flatten())
                train_labels.extend(labels.cpu().numpy().flatten())
            
            train_metrics = compute_metrics(train_labels, train_preds)
            print(f"Epoch {epoch+1}, Train Loss: {train_loss / len(loaders['train']):.4f}")
            print(f"Train Metrics: AUC={train_metrics['AUC']:.3f}, PRC={train_metrics['PRC']:.3f}, "
                  f"Acc={train_metrics['Accuracy']:.3f}, F1={train_metrics['F1']:.3f}, "
                  f"Precision={train_metrics['Precision']:.3f}, Recall={train_metrics['Recall']:.3f}")
        
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for seq1, seq2, labels in loaders['val']:
                seq1, seq2, labels = seq1.to(device), seq2.to(device), labels.to(device)
                prof_out, count_out, class_out = model(seq1, seq2)
                loss = combined_loss(prof_out, count_out, class_out, labels, model_params['counts_loss_weight'])
                val_loss += loss.item()
                val_preds.extend(class_out.cpu().numpy().flatten())
                val_labels.extend(labels.cpu().numpy().flatten())
        
        val_metrics = compute_metrics(val_labels, val_preds)
        print(f"Validation Loss: {val_loss / len(loaders['val']):.4f}")
        print(f"Validation Metrics: AUC={val_metrics['AUC']:.3f}, PRC={val_metrics['PRC']:.3f}, "
              f"Acc={val_metrics['Accuracy']:.3f}, F1={val_metrics['F1']:.3f}, "
              f"Precision={val_metrics['Precision']:.3f}, Recall={val_metrics['Recall']:.3f}")
        plot_roc_prc(task_name, val_labels, val_preds, 'val')
        
        test_preds, test_labels = [], []
        with torch.no_grad():
            for seq1, seq2, labels in loaders['test']:
                seq1, seq2, labels = seq1.to(device), seq2.to(device), labels.to(device)
                _, _, class_out = model(seq1, seq2)
                test_preds.extend(class_out.cpu().numpy().flatten())
                test_labels.extend(labels.cpu().numpy().flatten())
        
        test_metrics = compute_metrics(test_labels, test_preds)
        print(f"Test Metrics: AUC={test_metrics['AUC']:.3f}, PRC={test_metrics['PRC']:.3f}, "
              f"Acc={test_metrics['Accuracy']:.3f}, F1={test_metrics['F1']:.3f}, "
              f"Precision={test_metrics['Precision']:.3f}, Recall={test_metrics['Recall']:.3f}")
        plot_roc_prc(task_name, test_labels, test_preds, 'test')

# 主程序
if __name__ == "__main__":
    # 模型参数
    model_params = {
        'filters': 64,
        'n_dil_layers': 4,
        'counts_loss_weight': 0.1,
        'inputlen': 5000,
        'outputlen': 1000
    }
    args = type('Args', (), {'seed': 42, 'learning_rate': 0.001})()
    
    # 数据路径（需替换为实际路径）
    epi_train_seq1_path = '/public/home/shenyin_wsb_2606/Third/EP1000/5000/EPL_B.npz'
    epi_train_seq2_path = '/public/home/shenyin_wsb_2606/Third/EP1000/5000/EPR_B.npz'
    epi_test_seq1_path = '/public/home/shenyin_wsb_2606/Third/EP1000/5000/EPL_C.npz'
    epi_test_seq2_path = '/public/home/shenyin_wsb_2606/Third/EP1000/5000/EPR_C.npz'
    eei_train_seq1_path = '/public/home/shenyin_wsb_2606/Third/EE1000/5000/EEL_B.npz'
    eei_train_seq2_path = '/public/home/shenyin_wsb_2606/Third/EE1000/5000/EER_B.npz'
    eei_test_seq1_path = '/public/home/shenyin_wsb_2606/Third/EE1000/5000/EEL_C.npz'
    eei_test_seq2_path = '/public/home/shenyin_wsb_2606/Third/EE1000/5000/EER_C.npz'
    
    # 创建数据加载器
    task_loaders, task_val_labels, task_test_labels = create_multitask_data_loaders(
        epi_train_seq1_path, epi_train_seq2_path, epi_test_seq1_path, epi_test_seq2_path,
        eei_train_seq1_path, eei_train_seq2_path, eei_test_seq1_path, eei_test_seq2_path,
        batch_size=32, val_ratio=0.1, random_state=42
    )
    
    # 训练和评估
    train_and_evaluate(task_loaders, task_val_labels, task_test_labels, model_params, args)
