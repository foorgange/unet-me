# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from evaluate import compute_mIoU
from utils import save_model

# 模型训练函数，支持自定义 epoch、学习率、batch_size

def train(model, train_set, val_set, device, num_epochs=20, lr=1e-3, batch_size=4):
    # 数据加载器
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=2)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_miou = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        # 训练阶段
        for img, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            img, label = img.to(device), label.to(device)
            out = model(img)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f}")

        # 验证阶段
        model.eval()
        all_pred, all_label = [], []
        with torch.no_grad():
            for img, label in val_loader:
                img = img.to(device)
                pred = model(img).argmax(1).cpu()
                all_pred.append(pred)
                all_label.append(label)

        preds = torch.cat(all_pred)
        labels = torch.cat(all_label)
        miou = compute_mIoU(preds, labels, num_classes=6)
        print(f"[Epoch {epoch+1}] Validation mIoU: {miou:.4f}")

        # 保存最优模型
        if miou > best_miou:
            best_miou = miou
            save_model(model, 'best_model.pth')
            print("Best model saved!\n")
