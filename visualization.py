import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(train_losses, val_losses, val_accuracies):
    """
    绘制训练过程中的损失曲线和准确率曲线
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

def visualize_network_params(model):
    """
    可视化网络的所有重要参数
    """
    # 1. 可视化权重矩阵
    plt.figure(figsize=(15, 10))
    
    # 第一层权重
    plt.subplot(2, 2, 1)
    W1 = model.params['W1']
    plt.imshow(W1.T, cmap='viridis')
    plt.colorbar()
    plt.title('First Layer Weights (W1)')
    
    # 第二层权重
    plt.subplot(2, 2, 2)
    W2 = model.params['W2']
    plt.imshow(W2.T, cmap='viridis')
    plt.colorbar()
    plt.title('Second Layer Weights (W2)')
    
    # 权重分布直方图
    plt.subplot(2, 2, 3)
    plt.hist(W1.flatten(), bins=50, alpha=0.5, label='W1')
    plt.hist(W2.flatten(), bins=50, alpha=0.5, label='W2')
    plt.title('Weight Distributions')
    plt.legend()
    
    # 偏置分布
    plt.subplot(2, 2, 4)
    b1 = model.params['b1']
    b2 = model.params['b2']
    plt.hist(b1.flatten(), bins=20, alpha=0.5, label='b1')
    plt.hist(b2.flatten(), bins=20, alpha=0.5, label='b2')
    plt.title('Bias Distributions')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('network_params.png')
    plt.close()
    
    # 2. 可视化批量归一化参数
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    gamma = model.params['gamma1']
    plt.hist(gamma.flatten(), bins=20)
    plt.title('Batch Norm Gamma Distribution')
    
    plt.subplot(1, 2, 2)
    beta = model.params['beta1']
    plt.hist(beta.flatten(), bins=20)
    plt.title('Batch Norm Beta Distribution')
    
    plt.tight_layout()
    plt.savefig('batch_norm_params.png')
    plt.close()
    
    # 3. 打印参数统计信息
    print("\n网络参数统计信息:")
    print(f"W1: shape={W1.shape}, mean={np.mean(W1):.4f}, std={np.std(W1):.4f}")
    print(f"W2: shape={W2.shape}, mean={np.mean(W2):.4f}, std={np.std(W2):.4f}")
    print(f"b1: shape={b1.shape}, mean={np.mean(b1):.4f}, std={np.std(b1):.4f}")
    print(f"b2: shape={b2.shape}, mean={np.mean(b2):.4f}, std={np.std(b2):.4f}")
    print(f"gamma1: shape={gamma.shape}, mean={np.mean(gamma):.4f}, std={np.std(gamma):.4f}")
    print(f"beta1: shape={beta.shape}, mean={np.mean(beta):.4f}, std={np.std(beta):.4f}") 