import argparse
import torch
from dataset import VaihingenDataset
from model import UNet
from train import train
from utils import load_model

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--data_dir', type=str, default='./dates/vaihingen')
    parser.add_argument('--model_path', type=str, default='best_model.pth')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 设置图像和标签路径
    img_dir = f'{args.data_dir}/image'
    label_dir = f'{args.data_dir}/label'

    # 加载数据集
    train_set = VaihingenDataset(img_dir, label_dir, transform=True)
    val_set = VaihingenDataset(img_dir, label_dir, transform=False)

    # 初始化模型
    model = UNet(n_classes=6).to(device)

    if args.mode == 'train':
        train(model, train_set, val_set, device, num_epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    elif args.mode == 'eval':
        model = load_model(model, args.model_path, device)
        print("模型加载完成，后续可添加测试代码。")

if __name__ == "__main__":
    main()
