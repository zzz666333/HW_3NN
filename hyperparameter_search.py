import numpy as np
from model import TwoLayerNN
from utils import compute_accuracy
import os

def save_training_state(model, train_losses, val_losses, val_accuracies, 
                       current_params, current_epoch, current_batch):
    """保存训练状态"""
    state = {
        'model_params': model.params,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'current_params': current_params,
        'current_epoch': current_epoch,
        'current_batch': current_batch
    }
    np.save('training_state.npy', state)
    print(f"训练状态已保存: epoch={current_epoch}, batch={current_batch}")

def load_training_state():
    """加载训练状态"""
    if os.path.exists('training_state.npy'):
        state = np.load('training_state.npy', allow_pickle=True).item()
        print(f"加载训练状态: epoch={state['current_epoch']}, batch={state['current_batch']}")
        return state
    return None

def learning_rate_decay(initial_lr, epoch, decay_rate=0.95):
    """
    学习率衰减函数
    """
    return initial_lr * (decay_rate ** epoch)

def train_model(model, x_train, y_train, x_val, y_val, learning_rate, epochs, batch_size, 
                current_params=None, resume_epoch=0, resume_batch=0):
    best_val_acc = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # 尝试加载之前的训练状态
    state = load_training_state()
    if state is not None and current_params == state['current_params']:
        model.params = state['model_params']
        train_losses = state['train_losses']
        val_losses = state['val_losses']
        val_accuracies = state['val_accuracies']
        resume_epoch = state['current_epoch']
        resume_batch = state['current_batch']
        best_val_acc = max(val_accuracies) if val_accuracies else 0
        print(f"从epoch {resume_epoch}恢复训练")
    
    for epoch in range(resume_epoch, epochs):
        # 学习率衰减
        current_lr = learning_rate_decay(learning_rate, epoch)
        
        # 打乱训练数据
        shuffle_idx = np.random.permutation(len(x_train))
        x_train_shuffled = x_train[shuffle_idx]
        y_train_shuffled = y_train[shuffle_idx]
        
        # 训练一个epoch
        epoch_losses = []
        start_batch = resume_batch if epoch == resume_epoch else 0
        
        # 设置为训练模式
        model.set_mode('train')
        
        for i in range(start_batch, len(x_train), batch_size):
            x_batch = x_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]
            
            probs, cache = model.forward(x_batch)
            loss = model.compute_loss(probs, y_batch)
            grads = model.backward(probs, y_batch, cache)
            model.update_params(grads, current_lr)
            epoch_losses.append(loss)
            
            # 每100个batch保存一次状态
            if (i // batch_size) % 100 == 0:
                save_training_state(model, train_losses, val_losses, val_accuracies,
                                  current_params, epoch, i)
        
        # 计算验证集性能
        model.set_mode('test')  # 切换到测试模式
        val_probs, _ = model.forward(x_val)
        val_pred = np.argmax(val_probs, axis=1)
        val_acc = compute_accuracy(val_pred, y_val)
        val_loss = model.compute_loss(val_probs, y_val)
        model.set_mode('train')  # 切换回训练模式
        
        avg_epoch_loss = np.mean(epoch_losses)
        train_losses.append(avg_epoch_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # 只打印每个epoch的结果
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_epoch_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            np.save("checkpoint.npy", model.params)
            print(f"新的最佳验证准确率: {best_val_acc:.4f}")
        
        # 重置resume_batch
        resume_batch = 0
    
    return best_val_acc, train_losses, val_losses, val_accuracies

def hyperparameter_search(x_train, y_train, x_val, y_val):
    hidden_sizes = [256, 512]  # 减少隐藏层大小的选择
    learning_rates = [5e-3, 1e-3]  # 两个学习率
    reg_strengths = [1e-4]  # 固定使用1e-4的正则化强度
    dropout_rates = [0.5]  # 固定使用0.5的dropout率
    
    best_acc = 0
    best_params = {}
    best_model = None
    best_train_losses = []
    best_val_losses = []
    best_val_accuracies = []
    
    # 初始化搜索状态
    search_state = {
        'best_acc': 0,
        'best_params': {},
        'searched_params': []
    }
    
    # 尝试加载之前的搜索状态
    if os.path.exists('search_state.npy'):
        loaded_state = np.load('search_state.npy', allow_pickle=True).item()
        best_acc = loaded_state['best_acc']
        best_params = loaded_state['best_params']
        search_state = loaded_state
        print(f"从之前的搜索恢复，当前最佳准确率: {best_acc:.4f}")
    
    # 初始化一个默认模型
    default_model = TwoLayerNN(input_size=3072, 
                             hidden_size=hidden_sizes[0], 
                             output_size=10,
                             reg=reg_strengths[0],
                             dropout_rate=dropout_rates[0])
    default_train_losses = []
    default_val_losses = []
    default_val_accuracies = []
    
    for hidden_size in hidden_sizes:
        for lr in learning_rates:
            for reg in reg_strengths:
                for dropout_rate in dropout_rates:
                    current_params = {
                        'hidden_size': hidden_size,
                        'learning_rate': lr,
                        'reg': reg,
                        'dropout_rate': dropout_rate
                    }
                    
                    # 如果这个参数组合已经搜索过，跳过
                    if current_params in search_state['searched_params']:
                        continue
                    
                    print(f"\n开始训练: hidden_size={hidden_size}, lr={lr}")
                    
                    model = TwoLayerNN(input_size=3072, 
                                     hidden_size=hidden_size, 
                                     output_size=10,
                                     reg=reg,
                                     dropout_rate=dropout_rate)
                    
                    val_acc, train_losses, val_losses, val_accuracies = train_model(
                        model, x_train, y_train, x_val, y_val, lr, epochs=100, batch_size=32,
                        current_params=current_params)
                    
                    # 更新搜索状态
                    search_state['best_acc'] = max(best_acc, val_acc)
                    search_state['best_params'] = best_params if best_acc > val_acc else current_params
                    search_state['searched_params'].append(current_params)
                    
                    # 保存搜索状态
                    np.save('search_state.npy', search_state)
                    
                    # 如果是第一个模型或者准确率更高，更新最佳模型
                    if best_model is None or val_acc > best_acc:
                        best_acc = val_acc
                        best_params = current_params
                        best_model = model
                        best_train_losses = train_losses
                        best_val_losses = val_losses
                        best_val_accuracies = val_accuracies
                        print(f"新的最佳参数组合: {best_params}")
    
    # 确保返回一个模型
    if best_model is None:
        # 如果没有找到更好的模型，返回默认模型
        best_model = default_model
        best_params = {
            'hidden_size': hidden_sizes[0],
            'learning_rate': learning_rates[0],
            'reg': reg_strengths[0],
            'dropout_rate': dropout_rates[0]
        }
        best_train_losses = default_train_losses
        best_val_losses = default_val_losses
        best_val_accuracies = default_val_accuracies
    
    return best_params, best_model, best_train_losses, best_val_losses, best_val_accuracies 