import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as T
import random

# 颜色映射表：RGB颜色 -> 类别编号
PALETTE = {
    (255, 255, 255): 0,  # ImSurf
    (0, 0, 255): 1,      # Building
    (0, 255, 255): 2,    # LowVeg
    (0, 255, 0): 3,      # Tree
    (255, 204, 0): 4,    # Car
    (255, 0, 0): 5       # Clutter
}

def mask_to_class(mask):
    """
    将 RGB 掩码图像转换为整数标签矩阵
    每个像素点 RGB 匹配 PALETTE 中的某一类，映射为 0~5 的整数标签
    """
    mask = np.array(mask)  # (H, W, 3)
    label_mask = np.zeros(mask.shape[:2], dtype=np.int64)  # 初始化 label map

    for rgb, idx in PALETTE.items():
        matches = np.all(mask == rgb, axis=-1)  # 找出所有匹配某种颜色的位置
        label_mask[matches] = idx               # 设置为对应类别索引

    return label_mask  # 返回的是整数标签图 (H, W)


class VaihingenDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=True):
        """
        image_dir: 图像文件夹路径
        label_dir: 标签（掩码）文件夹路径
        transform: 是否应用数据增强（翻转/旋转）
        """
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像（RGB）和标签掩码图像（RGB）
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = Image.open(self.label_paths[idx]).convert('RGB')

        # 转为 numpy 并进行颜色到类别的映射
        img = np.array(img)
        label = mask_to_class(label)  # 得到标签图 (H, W)，值域在 0~5

        # ---------------------------
        # 数据增强（3种：左右翻转，上下翻转，旋转）
        # ---------------------------
        if self.transform:
            if random.random() > 0.5:
                img = np.fliplr(img).copy()
                label = np.fliplr(label).copy()
            if random.random() > 0.5:
                img = np.flipud(img).copy()
                label = np.flipud(label).copy()
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                img = np.rot90(img, k=angle // 90).copy()
                label = np.rot90(label, k=angle // 90).copy()

        # ---------------------------
        # resize 图像 & 标签至 256×256
        # 注意：标签需要使用 NEAREST 插值避免数值污染
        # ---------------------------
        img = Image.fromarray(img).resize((256, 256))
        label = Image.fromarray(label.astype(np.uint8)).resize((256, 256), Image.NEAREST)

        # ---------------------------
        # 转为 Tensor 格式
        # ---------------------------
        img = T.ToTensor()(img)  # 归一化到 [0,1]，shape: (3, 256, 256)
        label = torch.from_numpy(np.array(label)).long()  # shape: (256, 256)，类型为 long

        return img, label
