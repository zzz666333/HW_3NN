import pickle
import numpy as np
import os

#读取 CIFAR-10 的 .pkl 文件格式，返回字典。
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
#加载一个batch，并作归一化处理
def load_cifar10_batch(batch_path):
    batch = unpickle(batch_path)
    data = batch[b'data'] / 255.0  # Normalize
    labels = np.array(batch[b'labels'])
    return data, labels
#加载 CIFAR-10 的前五个训练批次（共 50000 张图），和一个测试集（10000 张图），统一格式
def load_cifar10_dataset(data_dir='cifar-10-batches-py'):
    x_train, y_train = [], []
    for i in range(1, 6):
        data, labels = load_cifar10_batch(os.path.join(data_dir, f'data_batch_{i}'))
        x_train.append(data)
        y_train.append(labels)
    x_train = np.vstack(x_train)
    y_train = np.hstack(y_train)
    x_test, y_test = load_cifar10_batch(os.path.join(data_dir, 'test_batch'))
    return x_train, y_train, x_test, y_test
