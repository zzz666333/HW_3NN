import numpy as np
from utils import one_hot_encode

class TwoLayerNN:
    #定义网络结构输入层 → 隐藏层 → 输出层
    #每一层都有W权重和b偏置
    #权重初始化为高斯随机小数（避免初始梯度为0），偏置为0
    #reg 是 L2 正则化项的系数
    def __init__(self, input_size, hidden_size, output_size, reg=1e-3, dropout_rate=0.5):
        self.params = {
            'W1': np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size),  # He初始化
            'b1': np.zeros((1, hidden_size)),
            'W2': np.random.randn(hidden_size, output_size) * np.sqrt(2.0/hidden_size),
            'b2': np.zeros((1, output_size)),
            'gamma1': np.ones((1, hidden_size)),  # 批量归一化参数
            'beta1': np.zeros((1, hidden_size)),
        }
        self.reg = reg
        self.dropout_rate = dropout_rate
        self.mode = 'train'  # 控制dropout和BN的模式
        
        # 初始化Batch Normalization的running mean和var
        self.running_mean = np.zeros((1, hidden_size))
        self.running_var = np.ones((1, hidden_size))
    #ReLU激活函数
    def relu(self, x):
        return np.maximum(0, x)
    #softmax函数
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    #向前传播
    def batch_norm_forward(self, x, gamma, beta):
        """
        批量归一化前向传播
        """
        if self.mode == 'train':
            # 计算均值和方差
            mu = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)
            
            # 更新running mean和var
            momentum = 0.9
            self.running_mean = momentum * self.running_mean + (1 - momentum) * mu
            self.running_var = momentum * self.running_var + (1 - momentum) * var
            
            # 标准化
            x_normalized = (x - mu) / np.sqrt(var + 1e-5)
            
            # 缩放和平移
            out = gamma * x_normalized + beta
            
            # 保存中间结果供反向传播使用
            self.bn_cache = (x, x_normalized, mu, var, gamma, beta)
            
        else:  # test mode
            # 使用训练时计算的移动平均
            mu = self.running_mean
            var = self.running_var
            
            x_normalized = (x - mu) / np.sqrt(var + 1e-5)
            out = gamma * x_normalized + beta
            
        return out

    def batch_norm_backward(self, dout, cache):
        """
        批量归一化反向传播
        """
        x, x_normalized, mu, var, gamma, beta = cache
        N, D = dout.shape
        
        # 计算梯度
        dgamma = np.sum(dout * x_normalized, axis=0, keepdims=True)
        dbeta = np.sum(dout, axis=0, keepdims=True)
        
        # 计算dx_normalized
        dx_normalized = dout * gamma
        
        # 计算dvar
        dvar = np.sum(dx_normalized * (x - mu) * (-0.5) * (var + 1e-5)**(-1.5), axis=0, keepdims=True)
        
        # 计算dmu
        dmu = np.sum(dx_normalized * (-1) / np.sqrt(var + 1e-5), axis=0, keepdims=True) + \
              dvar * np.sum(-2 * (x - mu), axis=0, keepdims=True) / N
        
        # 计算dx
        dx = dx_normalized / np.sqrt(var + 1e-5) + \
             dvar * 2 * (x - mu) / N + \
             dmu / N
        
        return dx, dgamma, dbeta

    def dropout_forward(self, x, p):
        """
        Dropout前向传播
        """
        if self.mode == 'train':
            mask = (np.random.rand(*x.shape) < p) / p
            out = x * mask
            self.dropout_cache = mask
        else:
            out = x
        return out

    def dropout_backward(self, dout):
        """
        Dropout反向传播
        """
        if self.mode == 'train':
            dx = dout * self.dropout_cache
        else:
            dx = dout
        return dx

    def forward(self, x):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        gamma1, beta1 = self.params['gamma1'], self.params['beta1']

        # 第一层
        h1 = x @ W1 + b1
        h1 = self.batch_norm_forward(h1, gamma1, beta1)
        h1 = self.relu(h1)
        h1 = self.dropout_forward(h1, self.dropout_rate)

        # 第二层
        out = self.softmax(h1 @ W2 + b2)

        cache = (x, h1)
        return out, cache
    #损失函数(带正则项)
    def compute_loss(self, probs, y_true):
        N = y_true.shape[0]
        corect_logprobs = -np.log(probs[range(N), y_true] + 1e-8)
        #交叉熵损失：衡量分类概率输出与真实标签的距离
        data_loss = np.sum(corect_logprobs) / N
        #正则化：防止过拟合，惩罚权重太大
        reg_loss = 0.5 * self.reg * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2))
        return data_loss + reg_loss
    #反向传播(手动推导梯度)
    def backward(self, probs, y_true, cache):
        x, h1 = cache
        grads = {}
        N = x.shape[0]
        y_true_one_hot = one_hot_encode(y_true, probs.shape[1])
    
        dout = (probs - y_true_one_hot) / N
        #用链式法则计算梯度
        grads['W2'] = h1.T @ dout + self.reg * self.params['W2']
        grads['b2'] = np.sum(dout, axis=0, keepdims=True)
        
        dh1 = dout @ self.params['W2'].T
        dh1 = self.dropout_backward(dh1)
        dh1[h1 <= 0] = 0  # ReLU 反向传播
        #计算隐藏层梯度
        grads['W1'] = x.T @ dh1 + self.reg * self.params['W1']
        grads['b1'] = np.sum(dh1, axis=0, keepdims=True)

        # 批量归一化梯度
        dh1, dgamma1, dbeta1 = self.batch_norm_backward(dh1, self.bn_cache)

        grads['gamma1'] = dgamma1
        grads['beta1'] = dbeta1

        return grads
    #参数更新
    def update_params(self, grads, learning_rate):
        for key in self.params:
            self.params[key] -= learning_rate * grads[key]
    #梯度下降法（SGD）：用学习率乘以每层梯度，更新网络参数
    def set_mode(self, mode):
        """
        设置模型模式：'train' 或 'test'
        """
        assert mode in ['train', 'test']
        self.mode = mode