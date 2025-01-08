import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
from torchvision import transforms
from P2_dataloader import SegmentationDataset
from P2_model_A import FCN32s
from collections import OrderedDict

# 定義 mean IoU 計算函數
def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
        print('class #%d : %1.5f' % (i, iou))
    print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou

# 訓練函數
def train_model(num_epochs=100, batch_size=4, learning_rate=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加載模型
    model_type = 'resnet101'
    model = FCN32s(n_class=7).to(device)

    # 定義 loss 和 optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,weight_decay=0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 定義學習率調整器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-6, T_max=100)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)      
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=1e-6)

    transforms_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transforms_validation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加載訓練和驗證數據集
    train_dataset = SegmentationDataset(img_dir='../hw1_data/p2_data/train/',
                                        transform=transforms_train)
    val_dataset = SegmentationDataset(img_dir='../hw1_data/p2_data/validation/',
                                      transform=transforms_validation)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    best_iou = 0.0  # 保存最佳驗證 IoU

    # 開始訓練
    print('training start!!')
    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            loss=0
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)


            # loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        current_lr = optimizer.param_groups[0]['lr']

        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {avg_train_loss:.4f} , lr: {current_lr}')
        # 調整學習率
        scheduler.step()
        # 每個 epoch 後進行驗證
        if epoch % 5 == 0 and epoch!=0:
            model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    outputs = model(images)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_preds.append(preds)
                    all_labels.append(masks.numpy())

            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            mean_iou = mean_iou_score(all_preds, all_labels)

            # 保存最佳模型
            if mean_iou > best_iou:
                best_iou = mean_iou
                if not os.path.exists('checkpoint'):
                    os.makedirs('checkpoint')
                torch.save(model.state_dict(), 'checkpoint/best_fcn32_model.pth')
                print(f"New best model saved with IoU: {best_iou:.4f}")




    print(f" best  saved  IoU: {best_iou:.4f}")

    print("Training complete")

# def set_seed(seed):
#     ''' set random seeds '''
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#         torch.cuda.manual_seed(seed)
if __name__ == "__main__":
    # 使用 argparse 來讓用戶選擇參數
    parser = argparse.ArgumentParser(description="Training DeepLabV3Model model for semantic segmentation.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for training")
    # set_seed(9527)

    args = parser.parse_args()

    # 開始訓練
    train_model(num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
