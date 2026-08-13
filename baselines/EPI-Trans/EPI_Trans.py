# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 21:00:50 2025

@author: 123
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Dense, Concatenate, BatchNormalization, Activation, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from transfomer import Transformer_Merged

# 模型超参数
merged_n_heads = 9
merged_feed_forward_size = 256
merged_encoder_stack = 1
en_pool_size = 15
pr_pool_size = 10
en_strides = en_pool_size
pr_strides = pr_pool_size
en_kernal_size = 80
pr_kernal_size = 61
num_filters = 72
model_dim = 100

def get_model():
    sequence_input1 = Input(shape=(5000, 4))
    sequence_input2 = Input(shape=(5000, 4))

    enhancer_conv_layer = Conv1D(filters=num_filters,
                                 kernel_size=en_kernal_size,
                                 padding="valid",
                                 activation='relu')(sequence_input1)

    enhancer_max_pool_layer = MaxPooling1D(pool_size=en_pool_size, strides=en_strides)(enhancer_conv_layer)

    promoter_conv_layer = Conv1D(filters=num_filters,
                                 kernel_size=pr_kernal_size,
                                 padding="valid",
                                 activation='relu')(sequence_input2)

    promoter_max_pool_layer = MaxPooling1D(pool_size=pr_pool_size, strides=pr_strides)(promoter_conv_layer)

    # 合并 enhancer 和 promoter
    merge1 = Concatenate(axis=1)([enhancer_max_pool_layer, promoter_max_pool_layer])

    merge2 = Dense(23016)(merge1)
    
    bn = BatchNormalization()(merge2)
    dt = Dropout(0.5)(bn)

    transformer1 = Transformer_Merged(encoder_stack=merged_encoder_stack,
                                     feed_forward_size=merged_feed_forward_size,
                                     n_heads=merged_n_heads,
                                     model_dim=model_dim)

    trf = transformer1(dt)

    Gmaxpool = GlobalMaxPooling1D()(trf)

    merge4 = Dense(50)(Gmaxpool)

    bn2 = BatchNormalization()(merge4)
    acti = Activation('relu')(bn2)

    preds = Dense(1, activation='sigmoid')(acti)

    model = Model([sequence_input1, sequence_input2], preds)

    return model

# F1 分数计算函数
def f1(y_true, y_pred):
    TP = K.sum(K.cast(K.equal(y_true, 1) & K.equal(K.round(y_pred), 1), 'float32'))
    FP = K.sum(K.cast(K.equal(y_true, 0) & K.equal(K.round(y_pred), 1), 'float32'))
    FN = K.sum(K.cast(K.equal(y_true, 1) & K.equal(K.round(y_pred), 0), 'float32'))
    TN = K.sum(K.cast(K.equal(y_true, 0) & K.equal(K.round(y_pred), 0), 'float32'))

    P = TP / (TP + FP + K.epsilon())
    R = TP / (TP + FN + K.epsilon())
    F1 = 2 * P * R / (P + R + K.epsilon())
    return F1

# 训练函数
def training(model):
    print('Loading data...')
    SEQ_LEN = 5000  # 更新为 5000
    enhancer_shape = (-1, 5000, 4)  # 更新形状

    # 加载数据
    seq1 = np.load('/public/home/shenyin_wsb_2606/Third/EP1000/5000/EPL_B.npz')
    seq2 = np.load('/public/home/shenyin_wsb_2606/Third/EP1000/5000/EPR_B.npz')

    # 准备数据
    label = seq1['label'].shape[0]
    np.random.seed(label)
    rand_index = np.arange(label)
    np.random.shuffle(rand_index)
    label = seq1['label'][rand_index]
    seq1 = seq1['sequence'].astype('float32').reshape(enhancer_shape)[rand_index]
    seq2 = seq2['sequence'].astype('float32').reshape(enhancer_shape)[rand_index]

    # 编译模型
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.0001),
                  metrics=['accuracy', f1])
    
    filename = 'EPI_Trans.h5'
    modelCheckpoint = ModelCheckpoint(filename, monitor='val_accuracy', save_best_only=True, mode='max')
    earlyStopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, mode='max')
    
    model.fit([seq1, seq2], label, epochs=50, batch_size=64,
              validation_split=0.1, callbacks=[modelCheckpoint, earlyStopping])

# 加载测试数据
def load_test_data():
    print('Loading test data...')
    SEQ_LEN = 5000  # 更新为 5000
    enhancer_shape = (-1, 5000, 4)  # 更新形状

    # 加载测试数据
    seq1_test = np.load('/public/home/shenyin_wsb_2606/Third/EP1000/5000/EPL_C.npz')
    seq2_test = np.load('/public/home/shenyin_wsb_2606/Third/EP1000/5000/EPR_C.npz')

    # 提取数据和标签
    test_labels = seq1_test['label']
    seq1_test = seq1_test['sequence'].astype('float32').reshape(enhancer_shape)
    seq2_test = seq2_test['sequence'].astype('float32').reshape(enhancer_shape)

    return [seq1_test, seq2_test], test_labels

# 测试模型
def test_model(model_path):
    # 定义模型
    model = get_model()

    # 加载模型权重
    model.load_weights(model_path)

    # 加载测试数据
    test_data, test_labels = load_test_data()

    # 进行预测
    predictions = model.predict(test_data)
    predicted_labels = (predictions > 0.5).astype(int).flatten()

    # 计算评估指标
    accuracy = accuracy_score(test_labels, predicted_labels)
    f1_value = f1_score(test_labels, predicted_labels)
    auc = roc_auc_score(test_labels, predictions)
    aupr = average_precision_score(test_labels, predictions)

    # 打印结果
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1_value:.4f}")
    print(f"Test AUC: {auc:.4f}")
    print(f"Test AUPR: {aupr:.4f}")

# 主程序
if __name__ == "__main__":
    # 定义模型
    model = get_model()
    # 打印模型结构
    model.summary()
    # 训练模型
    training(model)
    # 测试模型
    model_path = 'EPI_Trans.h5'
    test_model(model_path)