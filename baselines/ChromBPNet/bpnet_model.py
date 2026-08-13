import numpy as np
from tensorflow.keras.backend import int_shape
from tensorflow.keras.layers import Input, Cropping1D, add, Conv1D, GlobalAvgPool1D, Dense, Flatten, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import tensorflow as tf
import random as rn
import os

os.environ['PYTHONHASHSEED'] = '0'

def getModelGivenModelOptionsAndWeightInits(args, model_params):
    # 默认参数（可通过model_params覆盖）
    conv1_kernel_size = 21
    n_dil_layers = int(model_params.get('n_dil_layers', 9))
    filters = int(model_params.get('filters', 64))
    counts_loss_weight = float(model_params.get('counts_loss_weight', 1.0))
    sequence_len = int(model_params.get('inputlen', 5000))
    
    print("参数：")
    print(f"filters: {filters}")
    print(f"n_dil_layers: {n_dil_layers}")
    print(f"conv1_kernel_size: {conv1_kernel_size}")
    print(f"counts_loss_weight: {counts_loss_weight}")
    
    # 设置随机种子以确保可重复性
    seed = args.seed
    np.random.seed(seed)
    tf.random.set_seed(seed)
    rn.seed(seed)
    
    # 定义两条序列的输入
    seq1_input = Input(shape=(4, sequence_len), name='seq1')  # 形状：(batch_size, 4, 5000)
    seq2_input = Input(shape=(4, sequence_len), name='seq2')  # 形状：(batch_size, 4, 5000)
    
    # 转置输入为(batch_size, sequence_len, 4)
    seq1 = Lambda(lambda x: tf.transpose(x, [0, 2, 1]), name='transpose_seq1')(seq1_input)  # 形状：(batch_size, 5000, 4)
    seq2 = Lambda(lambda x: tf.transpose(x, [0, 2, 1]), name='transpose_seq2')(seq2_input)  # 形状：(batch_size, 5000, 4)
    
    # 共享卷积主干
    def bpnet_backbone(x, filters, conv1_kernel_size, n_dil_layers, name_prefix):
        # 第一个无扩张的卷积
        x = Conv1D(filters,
                   kernel_size=conv1_kernel_size,
                   padding='valid',
                   activation='relu',
                   name=f'{name_prefix}_bpnet_1st_conv')(x)
        
        layer_names = [str(i) for i in range(1, n_dil_layers + 1)]
        for i in range(1, n_dil_layers + 1):
            conv_layer_name = f'{name_prefix}_bpnet_{layer_names[i-1]}conv'
            conv_x = Conv1D(filters,
                            kernel_size=3,
                            padding='valid',
                            activation='relu',
                            dilation_rate=2**i,
                            name=conv_layer_name)(x)
            
            x_len = int_shape(x)[1]
            conv_x_len = int_shape(conv_x)[1]
            assert (x_len - conv_x_len) % 2 == 0, "裁剪必须对称"
            
            x = Cropping1D((x_len - conv_x_len) // 2, name=f'{name_prefix}_bpnet_{layer_names[i-1]}crop')(x)
            x = add([conv_x, x], name=f'{name_prefix}_add_{layer_names[i-1]}')
        
        return x
    
    # 分别处理seq1和seq2，使用不同的名称前缀
    seq1_features = bpnet_backbone(seq1, filters, conv1_kernel_size, n_dil_layers, name_prefix='seq1')
    seq2_features = bpnet_backbone(seq2, filters, conv1_kernel_size, n_dil_layers, name_prefix='seq2')
    
    # 联合特征（例如，拼接）
    combined_features = Concatenate(name='combine_seqs')([seq1_features, seq2_features])
    
    # 全局平均池化以减少空间维度
    pooled_features = GlobalAvgPool1D(name='gap')(combined_features)
    
    # 任务特定的输出头
    epi_out = Dense(1, activation='sigmoid', name='epi_output')(pooled_features)
    eei_out = Dense(1, activation='sigmoid', name='eei_output')(pooled_features)
    
    # 定义模型
    model = Model(inputs=[seq1_input, seq2_input], outputs=[epi_out, eei_out])
    
    # 使用二元交叉熵损失编译模型
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss=['binary_crossentropy', 'binary_crossentropy'],
        loss_weights=[1.0, counts_loss_weight],
        metrics=['accuracy']
    )
    
    return model

def save_model_without_bias(model, output_prefix):
    # 占位函数（此模型无偏置项需要移除）
    return