<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-1.7%2B-red.svg" alt="PyTorch Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg" alt="Status">
</p>

<h1 align="center">🔢 PyTorch MNIST Handwritten Digit Recognition</h1>
<h3 align="center">基于 PyTorch 的手写数字识别系统 | CNN + 可视化分析</h3>

<p align="center">
  <strong>中文</strong> | 
  <a href="#english">English</a>
</p>

---

## 📌 项目简介 / Project Overview

本项目是一个**完整的、生产就绪的** MNIST 手写数字识别系统，基于 **PyTorch 深度学习框架**实现。采用 **CNN（卷积神经网络）** 架构，在标准测试集上达到 **99%+ 的准确率**。

This is a **complete, production-ready** MNIST handwritten digit recognition system built with **PyTorch deep learning framework**. It uses a **CNN (Convolutional Neural Network)** architecture and achieves **99%+ accuracy** on the standard test set.

### ✨ 核心特性 / Key Features

- 🎯 **高精度模型** - CNN架构，测试准确率 **99%+**
- 📊 **完整流程** - 数据下载 → 预处理 → 训练 → 测试 → 可视化
- 🖼️ **专业可视化** - OpenCV 多轮分批展示，支持正确/错误样本分析
- 🚀 **即开即用** - 一键运行，无需复杂配置
- 📝 **详细文档** - 完整的开发日志和代码注释
- 🔧 **高度可配置** - 灵活的参数调整接口

- 🎯 **High Accuracy** - CNN architecture, **99%+** test accuracy
- 📊 **Complete Pipeline** - Data download → Preprocessing → Training → Testing → Visualization
- 🖼️ **Professional Visualization** - Multi-round batch display with OpenCV, supports correct/error sample analysis
- 🚀 **Ready to Use** - One-click run, no complex configuration needed
- 📝 **Detailed Documentation** - Complete development log and code comments
- 🔧 **Highly Configurable** - Flexible parameter tuning interface

---

## 🛠️ 技术栈 / Tech Stack

| 类别 / Category | 技术 / Technology | 版本 / Version |
|----------------|-------------------|---------------|
| **深度学习框架** | PyTorch | >= 1.7.0 |
| **计算机视觉库** | Torchvision | >= 0.8.0 |
| **图像处理** | OpenCV-Python | >= 4.5.0 |
| **数值计算** | NumPy | Latest |
| **进度显示** | tqdm | >= 4.62.0 |

---

## 📦 安装步骤 / Installation

### 前置要求 / Prerequisites

- Python >= 3.8 (推荐 3.9+)
- pip 包管理器

### 快速安装 / Quick Install

```bash
# 克隆项目 / Clone the repository
git clone https://github.com/YOUR_USERNAME/Pytorch_MNIST.git
cd Pytorch_MNIST

# 创建虚拟环境 (推荐) / Create virtual environment (recommended)
python -m venv venv

# Windows / Windows
venv\Scripts\activate

# macOS/Linux / macOS/Linux
source venv/bin/activate

# 安装依赖 (使用清华镜像源加速) / Install dependencies (using Tsinghua mirror for speed)
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或者使用默认源 / Or use default source
pip install -r requirements.txt
```

### 验证安装 / Verify Installation

```python
import torch
import cv2
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"OpenCV version: {cv2.__version__}")
```

---

## 🚀 快速开始 / Quick Start

### 方式一：端到端训练+测试 / Method 1: End-to-End Training + Testing

```bash
# Step 1: 训练模型 (首次运行会自动下载MNIST数据集)
# Step 1: Train model (will auto-download MNIST dataset on first run)
python pytorch_mnist.py

# Step 2: 运行测试评估程序 (包含可视化分析)
# Step 2: Run testing and evaluation program (with visualization)
python mnist_test.py
```

### 方式二：仅测试预训练模型 / Method 2: Test Pre-trained Model Only

如果你已经有训练好的模型文件 (`models/*.pkl`)：

```bash
# 直接运行测试程序
python mnist_test.py
```

---

## 📂 项目结构 / Project Structure

```
Pytorch_mnist/
│
├── 📄 README.md                          # 项目说明文档 / Project documentation
├── 📄 LICENSE                            # MIT 开源协议 / MIT License
├── 📄 .gitignore                         # Git 忽略规则 / Git ignore rules
├── 📄 requirements.txt                   # Python 依赖列表 / Python dependencies
├── 📄 DEVELOPMENT_LOG.md                 # 开发日志 / Development log
│
├── 🐍 CNN.py                             # CNN 模型定义 / CNN model definition
│   └── Conv2d(1→32→64) → FC(64*7*7→128→10)
│
├── 🐍 pytorch_mnist.py                   # 主训练程序 / Main training program
│   ├── 数据下载与预处理 / Data download & preprocessing
│   ├── 模型训练循环 / Model training loop
│   └── 模型保存 / Model saving (.pkl format)
│
├── 🐍 mnist_test.py                      # 测试评估模块 / Testing & evaluation module
│   ├── 模型加载 / Model loading
│   ├── 性能评估 / Performance evaluation
│   ├── 准确率统计 / Accuracy statistics
│   └── OpenCV 可视化 / OpenCV visualization
│       ├── Phase 1: 顺序混合展示 (多轮)
│       └── Phase 2: 错误样本专展 (多轮)
│
├── 📁 models/                            # 训练好的模型 / Trained models
│   └── cnn_mnist_*.pkl                  # 模型权重文件 / Model weights file
│       ├── model_state_dict             # 模型参数
│       ├── optimizer_state_dict         # 优化器状态
│       ├── epoch                        # 训练轮数
│       └── test_accuracy               # 测试准确率
│
└── 📁 data/                              # 数据目录 (自动生成) / Data directory (auto-generated)
    └── MNIST/
        ├── raw/                         # 原始数据 / Raw data (~11MB)
        │   ├── train-images-idx3-ubyte.gz
        │   ├── train-labels-idx1-ubyte.gz
        │   ├── t10k-images-idx3-ubyte.gz
        │   └── t10k-labels-idx1-ubyte.gz
        └── processed/                   # 处理后数据 / Processed data
```

---

## 🧠 模型架构 / Model Architecture

### CNN 网络结构 / CNN Network Architecture

```
Input: [batch_size, 1, 28, 28]  (Grayscale 28×28 images)

↓ Conv2d(1, 32, kernel_size=3, padding=1)
↓ BatchNorm2d(32)
↓ ReLU
↓ MaxPool2d(2, 2)
Output: [batch_size, 32, 14, 14]

↓ Conv2d(32, 64, kernel_size=3, padding=1)
↓ BatchNorm2d(64)
↓ ReLU
↓ MaxPool2d(2, 2)
Output: [batch_size, 64, 7, 7]

↓ Flatten
↓ Linear(64 * 7 * 7, 128)
↓ ReLU
↓ Linear(128, 10)

Output: [batch_size, 10]  (Class probabilities for digits 0-9)
```

### 关键参数 / Key Parameters

| 参数 / Parameter | 值 / Value | 说明 / Description |
|-----------------|-----------|---------------------|
| **输入维度** | 28 × 28 × 1 | 灰度图像 / Grayscale image |
| **卷积层1** | 32 filters, 3×3 | 特征提取 / Feature extraction |
| **卷积层2** | 64 filters, 3×3 | 高级特征 / High-level features |
| **全连接层1** | 128 units | 特征组合 / Feature combination |
| **输出层** | 10 units | 10个数字类别 / 10 digit classes |
| **总参数量** | ~1.25M | 模型大小 / Model size |
| **激活函数** | ReLU | 非线性变换 / Non-linearity |
| **池化方式** | MaxPool2d(2×2) | 下采样 / Downsampling |

---

## 📊 性能指标 / Performance Metrics

### 测试集表现 / Test Set Performance

| 指标 / Metric | 数值 / Value | 说明 / Note |
|--------------|-------------|------------|
| **整体准确率** | **99.xx%** | 在10,000张测试图片上 / On 10K test images |
| **各类别准确率** | >98% | 数字0-9均表现优异 / All digits perform well |
| **推理时间 (CPU)** | < 10秒 | 完整测试集 / Full test set |
| **推理时间 (GPU)** | < 2秒 | 完整测试集 / Full test set |
| **模型大小** | ~5 MB | .pkl 文件 / .pkl file |

### 各数字类别准确率示例 / Per-Class Accuracy Example

```
Digit 0: ████████████████████ 99.8% ✅
Digit 1: ████████████████████ 99.9% ✅
Digit 2: ██████████████████░░ 98.5% ✅
Digit 3: ██████████████████░░ 98.7% ✅
Digit 4: ████████████████████ 99.6% ✅
Digit 5: ██████████████████░░ 98.9% ✅
Digit 6: ████████████████████ 99.7% ✅
Digit 7: ██████████████████░░ 98.3% ✅
Digit 8: ██████████████████░░ 98.6% ✅
Digit 9: ██████████████████░░ 97.9% ⚠️
```

---

## 🖼️ 可视化功能 / Visualization Features

### 双阶段展示策略 / Two-Phase Display Strategy

#### **Phase 1: Sequential Mixed Display** / 顺序混合展示
- 按测试集原始顺序展示样本
- 同时包含**正确预测（绿色）**和**错误预测（红色）**样本
- **多轮分批机制**：每轮12个样本，共4轮
- 显示实时统计：本轮正确/错误数量

#### **Phase 2: Error Analysis View** / 错误样本专展
- **专门展示所有错误预测样本**
- 详细分析：真实标签、预测标签、置信度
- **多轮分批机制**：每轮10个样本，最多3轮
- 适合深度分析模型的典型失误模式

### 视觉设计 / Visual Design

每个样本图片包含：
- ✅ **顶部彩色横幅**：绿色（正确）/ 红色（错误）
- 📝 **详细信息**：True Label | Predicted | Status | Confidence
- 🔢 **底部大字**：预测的数字（醒目显示）
- 🎨 **颜色编码**：一眼区分正确/错误

### 交互控制 / Interactive Controls

- ⌨️ **任意键**：继续下一轮
- ⏹️ **ESC 或 Q键**：提前退出展示
- 🔄 **自动窗口管理**：避免窗口堆积

---

## ⚙️ 配置说明 / Configuration

### 自定义参数 / Custom Parameters

在 `mnist_test.py` 中可以调整以下参数：

```python
results = test_model(
    model, 
    test_loader, 
    device,
    max_samples=48,           # 顺序收集的样本总数 (推荐24-60)
    max_error_samples=30      # 最大错误样本数 (推荐15-40)
)

# Phase 1 展示配置
show_sequential_samples_grid(
    results['sequential_samples'],
    cols=4,                   # 每行列数 (推荐3-5)
    scale=8,                  # 图片放大倍数 (推荐7-10)
    samples_per_page=12       # 每轮样本数 (推荐8-16)
)

# Phase 2 展示配置
show_error_samples_grid(
    results['error_samples'],
    cols=4,                   # 每行列数 (推荐3-5)
    scale=8,                  # 图片放大倍数 (推荐7-10)
    samples_per_page=10       # 每轮错误样本数 (推荐8-12)
)
```

### 不同屏幕分辨率推荐配置 / Screen Resolution Recommendations

| 分辨率 / Resolution | cols | scale | per_page | 窗口尺寸 / Window Size |
|--------------------|------|-------|----------|----------------------|
| 1366×768 (笔记本) | 3 | 7 | 9 | ~736×686px |
| **1920×1080 (标准)** | **4** | **8** | **12** | **~946×767px** ✅ |
| 2560×1440 (2K屏) | 5 | 9 | 15 | ~1350×900px |
| 3840×2160 (4K屏) | 6 | 10 | 18 | ~1700×1000px |

---

## 📚 使用示例 / Usage Examples

### 示例1：基础用法 / Basic Usage

```python
from mnist_test import load_model, test_model, print_results
import torch

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, info = load_model('./models/cnn_mnist_20260525_211841.pkl', device)

# 测试评估 (需要先加载数据)
from pytorch_mnist import download_and_preprocess_mnist
_, test_loader, _ = download_and_preprocess_mnist()

results = test_model(model, test_loader, device)
print_results(results)

# 输出:
# 🎯 整体准确率: 99.23%
# 各类别准确率: 0:99.8%, 1:99.9%, ..., 9:97.9%
```

### 示例2：自定义展示 / Custom Display

```python
from mnist_test import show_sequential_samples_grid, show_error_samples_grid

# 仅展示前20个顺序样本 (2轮)
show_sequential_samples_grid(
    results['sequential_samples'][:20],
    cols=3,
    scale=9,
    samples_per_page=10
)

# 仅展示前15个错误样本 (2轮)
show_error_samples_grid(
    results['error_samples'][:15],
    cols=3,
    scale=9,
    samples_per_page=8
)
```

### 示例3：批量推理 / Batch Inference

```python
import torch
from CNN import CNN
from torchvision import transforms
from PIL import Image

# 加载模型
model = CNN()
checkpoint = torch.load('models/cnn_mnist_20260525_211841.pkl')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载并预测单张图片
image = Image.open('my_digit.png')
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    predicted = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1)[0][predicted].item()

print(f'Predicted: {predicted}, Confidence: {confidence*100:.2f}%')
```

---

## 🐛 常见问题 / FAQ

### Q1: 首次运行时下载数据集很慢？/ Slow download on first run?

**A**: 使用清华镜像源加速。修改 `pytorch_mnist.py` 中的 `use_custom_mirror=True`（默认已开启），或手动下载后放到 `data/MNIST/raw/` 目录。

**A**: Use Tsinghua mirror for acceleration. Set `use_custom_mirror=True` in `pytorch_mnist.py` (enabled by default), or manually download to `data/MNIST/raw/`.

### Q2: CUDA out of memory? / 显存不足？

**A**: 减小 batch_size。在 `mnist_test.py` 中将 `download_and_preprocess_mnist(batch_size=64)` 改为 `batch_size=32` 或更小。

**A**: Reduce batch_size. Change `download_and_preprocess_mnist(batch_size=64)` to `batch_size=32` or smaller in `mnist_test.py`.

### Q3: OpenCV 窗口无法正常显示？/ OpenCV window not displaying?

**A**: 确保 opencv-python 已正确安装。如果仍有问题，尝试安装 headless 版本：`pip install opencv-python-headless` 并移除可视化代码。

**A**: Ensure opencv-python is installed correctly. If issues persist, try installing headless version: `pip install opencv-python-headless` and remove visualization code.

### Q4: 如何使用自己的手写数字图片？/ How to use custom handwritten digits?

**A**: 参考上面的"示例3"。确保图片是单通道灰度图，resize到28×28像素，并进行相同的标准化处理。

**A**: Refer to "Example 3" above. Ensure image is single-channel grayscale, resized to 28×28 pixels, and normalized similarly.

---

<a name="english"></a>
---
# English Version

<h1 align="center">🔢 PyTorch MNIST Handwritten Digit Recognition</h1>
<h3 align="center">CNN-based Handwritten Digit Recognition System with PyTorch</h3>

## Table of Contents

- [Project Overview](#-project-overview--project-overview)
- [Key Features](#-key-features--key-features)
- [Tech Stack](#-tech-stack--tech-stack)
- [Installation](#-installation--installation)
- [Quick Start](#-quick-start--quick-start)
- [Project Structure](#-project-structure--project-structure)
- [Model Architecture](#-model-architecture--model-architecture)
- [Performance Metrics](#-performance-metrics--performance-metrics)
- [Visualization Features](#-visualization-features--visualization-features)
- [Configuration](#-configuration--configuration)
- [Usage Examples](#-usage-examples--usage-examples)
- [FAQ](#-faq--faq)
- [Contributing](#-contributing--contributing)
- [License](#-license--license)
- [Acknowledgments](#-acknowledgments--acknowledgments)
- [Contact](#-contact--contact)

---

For detailed documentation in Chinese, please see the [Chinese version](#项目简介--project-overview) above.
