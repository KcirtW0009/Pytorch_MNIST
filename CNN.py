import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一个卷积层：输入1通道，输出32通道，卷积核大小3x3，填充1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 第一个批归一化层，对32通道进行归一化
        self.bn1 = nn.BatchNorm2d(32)
        # ReLU激活函数
        self.relu = nn.ReLU()
        # 最大池化层，窗口大小2x2，步长2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二个卷积层：输入32通道，输出64通道，卷积核大小3x3，填充1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 第二个批归一化层，对64通道进行归一化
        self.bn2 = nn.BatchNorm2d(64)
        # 第一个全连接层，输入维度64*7*7，输出128
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 第二个全连接层，输入128，输出10（类别数）
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 第一层卷积 -> 批归一化 -> ReLU激活 -> 最大池化
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        # 第二层卷积 -> 批归一化 -> ReLU激活 -> 最大池化
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        # 将特征图展平为一维向量
        x = x.view(-1, 64 * 7 * 7)
        # 第一个全连接层 -> ReLU激活
        x = self.fc1(x)
        x = self.relu(x)
        # 第二个全连接层，输出最终分类结果
        x = self.fc2(x)
        return x
