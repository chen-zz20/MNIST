# MNIST
## 项目简介
MNIST是机器学习中用于图像分类的广泛使用的数据集。它包含60000个训练样本和10000个测试样本。每个样本是784×1列向量，源于28×28像素的灰度图像。

## 文件结构
```
    .
    |—— data #MNIST数据集
    |—— jittor #jittor版本的网络
    |—— pytorch #pytorch版本的网络
    |—— train #保存的模型
    |—— log #tensorboard保存的文件地址
    README.md
    .gitignore
```

## 环境配置
### pytorch环境
```
cd .
cd pytorch
conda env create -f environment.yml
pip install -r requirements.txt
```

### jittor环境
```
cd .
cd jittor
conda env create -f environment.yml
pip install -r requirements.txt
```

## 运行程序
目前只调试好了pytorch框架的全部功能以及jittor框架的使用cpu训练的功能（gpu在作者的服务器上会报错），
运行命令：
```
cd pytorch 或者 cd jittor
python main.py
```