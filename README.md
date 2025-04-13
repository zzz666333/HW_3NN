# HW从零开始构建三层神经网络分类器，实现图像分类

```bash
.
├── model.py                  # 模型定义
├── hyperparameter_search.py  # 参数搜索脚本
├── utils.py                  # 工具函数
├── data_loader.py            # CIFAR-10加载与处理
├── visualization.py          # 训练过程与参数可视化
└── main.py                   # 项目入口，调用训练 + 测试
```

## 数据集
下载[CIFAR]https://www.cs.toronto.edu/~kriz/cifar.html中的CIFAR-10数据集，将解压后的数据放入文件夹cifar-10-batches-py，并将文件夹新建在主目录

## 训练与测试
```bash
main.py
```
根据提示进行选择即可："请选择运行模式 (1: 训练新模型, 2: 测试已有模型): "
