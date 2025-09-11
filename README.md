

---

# U-Net PyTorch 实现

## 项目概述

本项目是经典图像分割模型 U-Net 的 PyTorch 实现。U-Net 是一种专为生物医学图像分割而设计的卷积神经网络架构，由于其出色的性能和高效的结构，已被广泛应用于各种语义分割任务中，包括遥感图像分析、医学影像和常规照片分割等。

该代码库提供了一个结构清晰、易于理解和使用的 U-Net 模型，并附带了完整的训练和预测（推理）脚本，旨在帮助研究人员和开发者快速上手并将其应用于自己的数据集中。

## 主要特性

*   **经典的 U-Net 架构**: 完整实现了 U-Net 的编码器-解码器结构，包括下采样路径（收缩路径）和上采样路径（扩张路径），以及用于精确定位的跳跃连接（Skip Connections）。
*   **模块化设计**: 代码组织清晰，将模型结构、训练流程和工具函数分离开来，方便用户进行修改、扩展或集成到其他项目中。
*   **开箱即用的脚本**:
    *   `train.py`: 一个功能完整的训练脚本，支持从头开始训练或加载预训练权重继续训练。
    *   `predict.py`: 用于对单张或多张图像进行分割预测，并可视化结果。
*   **可配置性**: 训练参数（如学习率、批量大小、训练周期）和文件路径都可以通过配置文件或命令行参数轻松调整。
*   **支持多类别分割**: 该实现支持二分类和多分类的语义分割任务。

## 文件结构

```
unet-me/
├── src/
│   ├── __init__.py
│   ├── model.py         # U-Net 模型的核心定义
│   ├── train.py         # 训练脚本
│   ├── predict.py       # 推理（预测）脚本
│   ├── dataset.py       # 自定义数据集加载类
│   └── utils.py         # 工具函数，如绘图、数据处理等
├── .gitignore           # Git 忽略文件配置
└── README.md            # 本文档
```

## 安装

1.  **克隆代码库**:
    ```bash
    git clone https://github.com/foorgange/unet-me.git
    cd unet-me
    ```

2.  **创建并激活 Conda 环境 (推荐)**:
    ```bash
    conda create -n unet-me python=3.8
    conda activate unet-me
    ```

3.  **安装依赖**:
    项目依赖项通常包括 PyTorch, TorchVision, NumPy, Matplotlib 和 Pillow。您可以根据 `src` 目录下的 `import` 语句手动安装，或创建一个 `requirements.txt` 文件后通过 pip 安装。
    ```bash
    # 手动安装核心依赖
    pip install torch torchvision
    pip install numpy matplotlib pillow
    ```

## 使用方法

### 1. 数据集准备

为了使用 `dataset.py` 中的数据加载器，请将您的图像和对应的掩码（标签图）按以下结构存放：

```
/path/to/your/dataset/
├── images/          # 存放原始训练图像
│   ├── 1.png
│   ├── 2.png
│   └── ...
└── masks/           # 存放对应的分割掩码
    ├── 1.png
    ├── 2.png
    └── ...
```
**注意**: 请确保图像和其对应的掩码文件名相同。

### 2. 模型训练

通过运行 `train.py` 脚本来启动训练过程。您可以直接修改脚本内的参数，或通过命令行进行配置。

**训练示例**:

```bash
python src/train.py --epochs 50 --batch-size 4 --lr 0.001 --data-path /path/to/your/dataset
```

**重要的命令行参数**:

*   `--epochs`: 训练的总轮数。
*   `--batch-size`: 每个批次的图像数量。
*   `--lr`: 初始学习率。
*   `--data-path`: 数据集所在的根目录。
*   `--device`: 指定训练设备 (例如: `cuda` 或 `cpu`)。
*   `--weights`: (可选) 指定预训练模型权重的路径以继续训练。

训练过程中，模型的权重将定期保存在指定的目录中（默认为 `.`）。

### 3. 图像预测（推理）

训练完成后，使用 `predict.py` 脚本对新的图像进行分割。

**预测示例**:

```bash
python src/predict.py --weights /path/to/your/model_weights.pth --input /path/to/image.png --output /path/to/save/result.png
```

**重要的命令行参数**:

*   `--weights`: 指定已训练好的模型权重文件路径 (`.pth` 文件)。
*   `--input`: 需要进行分割的输入图像路径。
*   `--output`: 分割结果掩码的保存路径。
*   `--num-classes`: 模型的输出类别数，需要与训练时保持一致。

## U-Net 架构简介

U-Net 的结构呈 "U" 形，由两部分组成：

1.  **编码器 (Encoder / Contracting Path)**:
    *   由一系列的卷积层和池化层组成。
    *   负责从输入图像中提取层次化的特征。随着网络深度的增加，特征图的尺寸逐渐减小，但通道数（特征数量）逐渐增多。

2.  **解码器 (Decoder / Expansive Path)**:
    *   通过上采样（如转置卷积）操作，逐步将低分辨率的特征图恢复到原始图像尺寸。
    *   **跳跃连接 (Skip Connections)**: U-Net 的关键创新之一。它将编码器中对应层级的特征图直接拼接到解码器的上采样输出上。这使得解码器在重建分割图时，能够同时利用到底层的精细纹理信息和高层的语义信息，从而实现非常精确的像素级定位。

这种结构使得 U-Net 在数据量有限的情况下也能取得非常好的分割效果。

