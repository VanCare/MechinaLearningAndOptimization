#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt


MODEL = None


def get_model(in_dim,ls,activation):
    nn_architecture = []
    nn_architecture.append({"input_dim": int(in_dim), "output_dim": ls[0], "activation": activation})
    for i in range(len(ls)-1):
        nn_architecture.append({"input_dim": int(ls[i]), "output_dim": int(ls[i+1]), "activation": activation})
    nn_architecture.append({"input_dim": int(ls[-1]), "output_dim": 1, "activation": 'linear'})
    return nn_architecture


def linear(Z):
    copy_z = np.array(Z, copy=True)
    return copy_z


def linear_back(dA, Z):
    dZ = np.array(dA, copy=True)
    return dZ


def relu(Z):
    return np.maximum(0, Z)


def relu_back(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def sigmoid_back(dA, Z):
    sig = sigmoid(Z)
    dZ = dA * sig * (1 - sig)
    return dZ


def tanh(Z):
    return np.tanh(Z)


def tanh_back(dA, Z):
    return dA * (1 - tanh(Z) ** 2)


def leaky_relu(Z, a=0.01):
    copy_z = np.array(Z, copy=True)
    copy_z[copy_z < 0] = a * copy_z[copy_z < 0]
    return copy_z


def leaky_relu_back(dA, Z, a=0.01):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = dA[Z < 0] * a
    return dZ


def init_params(nn_architecture, seed=1964):
    param_values = []
    np.random.seed(seed)
    scale = 1.0
    for layer in nn_architecture:
        W = np.random.rand(layer['output_dim'], layer['input_dim']) * scale - (0.5 * scale)
        b = np.random.rand(layer['output_dim'], 1) * scale - (0.5 * scale)
        param_values.append({'W': W, 'b': b})
    return param_values


def single_layer_forward(A_prev, W_curr, b_curr, activation):
    if activation == 'relu':
        activation_func = relu
    elif activation == 'leaky_relu':
        activation_func = leaky_relu
    elif activation == 'linear':
        activation_func = linear
    elif activation == 'tanh':
        activation_func = tanh
    elif activation == 'sigmoid':
        activation_func = sigmoid
    else:
        raise Exception('Activation function not supported')
    U_curr = np.dot(W_curr, A_prev)
    Z_curr = U_curr + b_curr
    A_curr = activation_func(Z_curr)
    return Z_curr, A_curr


def forward_pass(X, param_values, nn_architecture):
    memory = [{'A': X, 'Z': None}]
    for i, layer in enumerate(nn_architecture):
        Z, A = single_layer_forward(memory[i]['A'], param_values[i]['W'], param_values[i]['b'],
                                    nn_architecture[i]['activation'])
        memory.append({'A': A, 'Z': Z})
        i += 1
    return memory


def squared_loss(y_hat, y):
    return (y_hat - y) ** 2 / 2


def get_cost_value(Y_hat, Y):
    m = Y_hat.shape[1]
    cost = 1 / (2 * m) * np.sum((Y_hat - Y) ** 2)
    return np.squeeze(cost)


def MSE_back(Y_hat, Y):
    return -(Y - Y_hat)


def single_layer_back(A_prev, Z_curr, W_curr, b_curr, dA_curr, activation):
    if activation == 'relu':
        back_activation_func = relu_back
    elif activation == 'leaky_relu':
        back_activation_func = leaky_relu_back
    elif activation == 'linear':
        back_activation_func = linear_back
    elif activation == 'tanh':
        back_activation_func = tanh_back
    elif activation == 'sigmoid':
        back_activation_func = sigmoid_back
    else:
        raise Exception('Activation function not supported')
    m= dA_curr.shape[1]
    dZ_curr = back_activation_func(dA_curr, Z_curr)
    db_curr = (1 / m) * np.sum(dZ_curr, axis=1, keepdims=True)
    dA_prev = np.dot(W_curr.T, dZ_curr)
    dW_curr = (1 / m) * np.dot(dZ_curr, A_prev.T)
    return dA_prev, db_curr, dW_curr


def back_pass(Y, memory, param_values, nn_architecture):
    grad_values = [dict() for x in range(len(nn_architecture))]
    Y_hat = memory[-1]['A']
    Y = Y.reshape(Y_hat.shape)
    dA_prev = MSE_back(Y_hat, Y)
    for i, layer in enumerate(nn_architecture):
        dA_curr = dA_prev
        layer_i = - i - 1
        dA_prev, db_curr, dW_curr = single_layer_back(memory[layer_i - 1]['A'],
                                                      memory[layer_i]['Z'],
                                                      param_values[layer_i]['W'],
                                                      param_values[layer_i]['b'],
                                                      dA_curr,
                                                      nn_architecture[layer_i]['activation'])
        grad_values[layer_i].update({'db': db_curr, 'dW': dW_curr})
    return grad_values


def update_params(grad_values, param_values, nn_architecture, learning_rate):
    for i, layer in enumerate(nn_architecture):
        param_values[i] = {'W': param_values[i]['W'] - grad_values[i]['dW'] * learning_rate,
                           'b': param_values[i]['b'] - grad_values[i]['db'] * learning_rate}
    return param_values.copy()


def train(X, Y, nn_architecture, epochs, learning_rate, x_std, x_mean, X_name, Y_name, seed=1964):
    history = {'loss': [],
               'params': None,
               'model': nn_architecture,
               'Normalization': [x_mean, x_std, X_name, Y_name,nn_architecture]}
    param_values = init_params(nn_architecture, seed)
    for epoch in range(epochs):
        memory = forward_pass(X, param_values, nn_architecture)
        Y_hat = memory[-1]['A']
        if epoch % 10 == 0:
            history['loss'].append(get_cost_value(Y_hat, Y))
        # history['params'].append(param_values)
        grad_values = back_pass(Y, memory, param_values, nn_architecture)
        param_values = update_params(grad_values, param_values, nn_architecture, learning_rate)
        history['params'] = param_values

    return history


def predict(model, X, Y=None):  # model(nn_结构,w,b,normalization数据。该数据来自于train函数的 return: history);X,Y为 ndarray.
    param_values = model['params']
    x_mean = model['Normalization'][0]
    x_std = model['Normalization'][1]
    x_name = model['Normalization'][2]
    y_name = model['Normalization'][3]
    nn_architecture = model['Normalization'][4]
    # 归一化X
    normalization = lambda x: (x - x_mean) / x_std
    X = np.apply_along_axis(normalization, 1, X).T #(4,1)
    memory = [{'A': X, 'Z': None}]
    for i, layer in enumerate(nn_architecture):
        Z, A = single_layer_forward(memory[i]['A'], param_values[i]['W'], param_values[i]['b'],
                                    nn_architecture[i]['activation'])
        memory.append({'A': A, 'Z': Z})
        i += 1
    if type(Y) == type(None):  # 预测模式
        s = memory[-1]['A']
        s2 = np.squeeze(s)
        return np.squeeze(memory[-1]['A'])
    else:  # 误差模式
        Y = Y.T
        return get_cost_value(memory[-1]['A'], Y)


def trans_predict(X):
    return float(predict(MODEL, [list(X)]))


# 写一个测试上述程序的主函数
if __name__ =='__main__':
    # 模拟人员操作前端——保存设置
    file_name = r"C:\Users\王凯\Desktop\机器学习相关\data\data\冷却塔数据\train.csv"
    training_ritio = 70 / 100
    independent_variable = ["a", "b", "c", "d"]
    dependent_variable = ["输出"]

    independent_variable_num = len(independent_variable)
    dependent_variable_num = len(dependent_variable)

    # 模拟人员操作前端——超参数调节
    learning_rate = 0.005
    number_of_hidden_nodes = [9, 3]
    activation_function = "relu"

    # 其余参数
    k = 5  # 5折交叉
    num_epochs = 2000  # 训练20轮
    weight_decay = 0  # 权重衰减
    batch_size = 32  # 小批量个数

    number_of_hidden_layers = len(number_of_hidden_nodes)

    # 数据集采集
    all_datas = pd.read_csv(file_name, encoding="gbk")
    try:
        all_datas = all_datas.astype("float32")
    except:
        print("文件中部分数据不是数字，无法训练。")
    all_datas = all_datas.sample(frac=1)
    all_datas = all_datas.reset_index()

    n_train = int(training_ritio * len(all_datas))

    # 预处理数据
    train_data = all_datas[0:n_train]
    test_data = all_datas[n_train:-1]

    # 标准化 df
    x_mean = np.mean(train_data[independent_variable].values, axis=0)
    x_std = np.std(train_data[independent_variable].values, axis=0)
    normalization = lambda x: (x - x_mean) / x_std
    train_data.loc[:, independent_variable] = np.apply_along_axis(func1d=normalization, axis=1,
                                                                  arr=train_data[independent_variable].values)
    # 训练集
    train_features = train_data[independent_variable].values
    train_labels = train_data[dependent_variable].values

    # 测试集
    test_features = test_data[independent_variable].values
    test_labels = test_data[dependent_variable].values

    # 模型
    nn_architecture = [
        {"input_dim": 4, "output_dim": 9, "activation": "leaky_relu"},
        {"input_dim": 9, "output_dim": 3, "activation": "leaky_relu"},
        {"input_dim": 3, "output_dim": 1, "activation": "linear"},
    ]
    history = train(X=np.transpose(train_features),
                    Y=np.transpose(train_labels.reshape((train_labels.shape[0], 1))),
                    nn_architecture=nn_architecture,
                    epochs=num_epochs,
                    learning_rate=0.01,
                    x_std=x_std,
                    x_mean=x_mean,
                    X_name=independent_variable,
                    Y_name=dependent_variable,
                    seed=1964)

    # plt.figure()
    # plt.plot(history['loss'])
    # plt.title('loss')
    # plt.yscale('log')
    # plt.show()

    print(history['loss'])
    loss_ = predict(history, test_features, test_labels)
    y = predict(history, test_features)
    print('loss:', loss_)
    print('y_predict:', y[:5])
    print('y:', test_labels[:5].T)
    print('y_predict:', y[5:10])
    print('y:', test_labels[5:10].T)
    print('y_predict:', y[10:15])
    print('y:', test_labels[10:15].T)