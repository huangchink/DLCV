import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import argparse
import os
import torchvision.transforms as transforms

from P1_model import ClassificationModel
from P1_dataloader import OfficeDataset
import logging

def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,  
        format='%(asctime)s - %(levelname)s - %(message)s', 
        handlers=[
            logging.FileHandler(log_file),  
            logging.StreamHandler()  
        ]
    )

# 定義計算準確率的函數
def calculate_accuracy(loader, model, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# t-SNE 可視化
def visualize_tsne(features, labels, epoch, stage, setting):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(features)
    
    # 確保 'output/' 目錄存在
    if not os.path.exists('output'):
        os.makedirs('output')

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title(f't-SNE visualization - {stage} Epoch {epoch}')
    plt.savefig(f'output/{setting}_tsne_epoch_{epoch}_{stage}.png')
    plt.close()

# 訓練和驗證
def train_model(setting='C', num_epochs=3, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加載模型
    model = ClassificationModel(setting=setting).to(device)
    
    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomResizedCrop((128,128)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        # transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.RandomRotation((-10, 10))  

    ])
    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加載數據集
    train_dataset = OfficeDataset(csv_file='../hw1_data/p1_data/office/train.csv',
                                  img_dir='../hw1_data/p1_data/office/train/', mode='train', transform=transform_train)
    val_dataset = OfficeDataset(csv_file='../hw1_data/p1_data/office/val.csv',
                                img_dir='../hw1_data/p1_data/office/val/', mode='train', transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 每20個epoch學習率下降50%
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-6, T_max=100)

    train_acc_list = []
    val_acc_list = []
    best_val_acc = 0.0  # 保存最佳驗證準確率
    logging.info('Training start!!!')
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train
        val_accuracy = calculate_accuracy(val_loader, model, device)

        train_acc_list.append(train_accuracy)
        val_acc_list.append(val_accuracy)

        # 獲取當前學習率
        current_lr = optimizer.param_groups[0]['lr']

        logging.info(f"Epoch [{epoch}/{num_epochs}] - Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%, LR: {current_lr:.6f}")

        # 如果目前的驗證準確率更高，保存模型
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            if not os.path.exists('checkpoint'):
                os.makedirs('checkpoint')
            if setting == 'C':
                torch.save(model.state_dict(), f'checkpoint/{setting}_best_model.pth')
            logging.info(f"Model best accuracy: {best_val_acc:.2f}%")

        # 學習率調整
        scheduler.step()

        # 畫 t-SNE 只在第一個和最後一個 epoch 並且是 setting C
        if setting == 'C' and (epoch == 1 or epoch == num_epochs):
            model.eval()
            features_list = []
            labels_list = []
            with torch.no_grad():
                for inputs, lbls in train_loader:
                    inputs = inputs.to(device)
                    
                    # 通過 backbone 前向傳播來獲取特徵（包括卷積層和 avgpool 層）
                    features = model.backbone(inputs)
                    
                    # 展平特徵以傳入 classifier
                    # outputs = torch.flatten(features, 1)                    
                    
                    # 將輸出轉換為 numpy 並存儲
                    features_list.append(features.cpu().numpy())
                    labels_list.append(lbls.numpy())

            features_list = np.concatenate(features_list)
            labels_list = np.concatenate(labels_list)
            visualize_tsne(features_list, labels_list, epoch, 'train', setting)

    # 畫出準確率曲線，並保存到 output 目錄
    if not os.path.exists('output'):
        os.makedirs('output')

    plt.figure()
    plt.plot(train_acc_list, label='Train Accuracy')
    plt.plot(val_acc_list, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig(f'output/{setting}_accuracy_curve.png')
    plt.close()

if __name__ == "__main__":
    # 使用 argparse 來讓用戶選擇參數
    parser = argparse.ArgumentParser(description="Training model with different settings.")
    parser.add_argument('--setting', type=str, default='C', help="The setting to use: A, B, C, D, E")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--log_file', type=str, default='output/train.log', help="Path to the log file")

    args = parser.parse_args()

    setup_logger(args.log_file)
    # 進行訓練
    train_model(setting=args.setting, num_epochs=args.epochs, batch_size=args.batch_size)
