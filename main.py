from data_loader import load_cifar10_dataset
from model import TwoLayerNN
from utils import compute_accuracy
from hyperparameter_search import hyperparameter_search
from visualization import plot_training_curves, visualize_network_params
import numpy as np
import os

def main():
    # 加载数据
    x_train, y_train, x_test, y_test = load_cifar10_dataset()
    
    # 打乱数据并划分验证集
    shuffle_idx = np.random.permutation(len(x_train))
    x_train, y_train = x_train[shuffle_idx], y_train[shuffle_idx]
    val_size = 5000
    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_train, y_train = x_train[val_size:], y_train[val_size:]
    
    # 超参数搜索
    print("开始超参数搜索...")
    best_params, best_model, train_losses, val_losses, val_accuracies = hyperparameter_search(
        x_train, y_train, x_val, y_val)
    
    print("\n最佳超参数:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # 可视化训练过程
    plot_training_curves(train_losses, val_losses, val_accuracies)
    
    # 可视化网络参数
    visualize_network_params(best_model)
    
    # 在测试集上评估
    test_probs, _ = best_model.forward(x_test)
    test_pred = np.argmax(test_probs, axis=1)
    test_acc = compute_accuracy(test_pred, y_test)
    print(f"\n测试集准确率: {test_acc:.4f}")

def test_model():
    # 加载数据
    x_train, y_train, x_test, y_test = load_cifar10_dataset()
    
    # 打乱数据并划分验证集
    shuffle_idx = np.random.permutation(len(x_train))
    x_train, y_train = x_train[shuffle_idx], y_train[shuffle_idx]
    val_size = 5000
    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_train, y_train = x_train[val_size:], y_train[val_size:]
    
    # 创建模型实例
    model = TwoLayerNN(input_size=3072, 
                      hidden_size=512,  # 使用最佳参数
                      output_size=10,
                      reg=1e-4,
                      dropout_rate=0.5)
    
    # 加载最佳模型参数
    if os.path.exists('checkpoint.npy'):
        model.params = np.load('checkpoint.npy', allow_pickle=True).item()
        print("已加载最佳模型参数")
    else:
        print("未找到保存的模型参数")
        return
    
    # 在验证集上测试
    model.set_mode('test')
    val_probs, _ = model.forward(x_val)
    val_pred = np.argmax(val_probs, axis=1)
    val_acc = compute_accuracy(val_pred, y_val)
    print(f"验证集准确率: {val_acc:.4f}")
    
    # 在测试集上测试
    test_probs, _ = model.forward(x_test)
    test_pred = np.argmax(test_probs, axis=1)
    test_acc = compute_accuracy(test_pred, y_test)
    print(f"测试集准确率: {test_acc:.4f}")
    
    # 可视化网络参数
    visualize_network_params(model)

if __name__ == "__main__":
    # 选择运行模式
    mode = input("请选择运行模式 (1: 训练新模型, 2: 测试已有模型): ")
    if mode == "1":
        main()
    elif mode == "2":
        test_model()
    else:
        print("无效的选择")
