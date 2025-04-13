import numpy as np

#把标签（如 [3, 2, 1]）转换成 one-hot 编码，用于交叉熵计算
def one_hot_encode(y, num_classes):
    """
    将标签 y 转换为 one-hot 编码
    y: shape (N,) int 标签
    返回: shape (N, num_classes) 的 one-hot 矩阵
    """
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

#计算预测准确率
#preds 是模型输出的类别预测，labels 是真实值，返回正确预测占比
def compute_accuracy(preds, labels):
    return np.mean(preds == labels)
