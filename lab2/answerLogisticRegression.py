import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
lr = 0.2  # 自动调参设定
wd = 0.005  # 自动调参设定


def predict(X, weight, bias):
    """
    使用输入的weight和bias，预测样本X是否为数字0。
    @param X: (n, d) 每行是一个输入样本。n: 样本数量, d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @return: (n,) 线性模型的输出，即wx+b
    """
    # TODO: YOUR CODE HERE
    return X@weight+bias

def sigmoid(x):
    out = np.zeros_like(x)
    pos_mask = (x >= 0)          # x >= 0 的位置
    neg_mask = (x < 0)           # x < 0 的位置
    
    out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[neg_mask])
    out[neg_mask] = exp_x / (1.0 + exp_x)
    
    return out

def logforpow(x):
    out = np.zeros_like(x)
    pos_mask = (x >= 0)
    neg_mask = (x < 0)

    out[pos_mask] = x[pos_mask] + np.log(1+np.exp(-x[pos_mask]))
    out[neg_mask] = np.log(1+np.exp(x[neg_mask]))
    return out

def step(X, weight, bias, Y):
    """
    单步训练, 进行一次forward、backward和参数更新
    @param X: (n, d) 每行是一个训练样本。 n: 样本数量， d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @param Y: (n,) 样本的label, 1表示为数字0, -1表示不为数字0
    @return:
        haty: (n,) 模型的输出, 为正表示数字为0, 为负表示数字不为0
        loss: (1,) 由交叉熵损失函数计算得到
        weight: (d,) 更新后的weight参数
        bias: (1,) 更新后的bias参数
    """
    # TODO: YOUR CODE HERE
    n = X.shape[0]

    haty = predict(X, weight, bias)
    prob = sigmoid(haty)

    Y01 = (Y + 1) / 2.0  # 原始 Y 是 -1/1 -> 转成 0/1

    cross_loss = np.mean(logforpow(-Y * haty))
    reg_loss =wd * np.sum(weight ** 2)
    total_loss = cross_loss + reg_loss

    grad_output = prob - Y01
    dw = (X.T @ grad_output) / n + 2*wd*weight
    db = np.mean(grad_output)

    new_weight = weight - lr * dw
    new_bias = bias - lr * db

    return haty, total_loss, new_weight, new_bias