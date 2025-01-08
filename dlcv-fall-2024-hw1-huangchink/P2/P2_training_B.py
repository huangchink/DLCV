import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
from torchvision import transforms
from P2_dataloader import SegmentationDataset, test_plot_Dataset
from P2_model_B import DeepLabV3Model
from PIL import Image
from collections import OrderedDict
import random

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
    model = DeepLabV3Model(model_type=model_type, num_classes=7).to(device)

    # 定義 loss 和 optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,weight_decay=0.0001)

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
    test_dataset = test_plot_Dataset(img_dir='../hw1_data/p2_data/validation/',
                                      transform=transforms_validation)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    plot_epochs = [1, 50, 100]
    plot_root = 'plot_output'
    os.makedirs(plot_root, exist_ok=True)
    test_plotset_loader = DataLoader(test_dataset, batch_size=3, shuffle=False)

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
            # loss = criterion(outputs['out'], masks)
            if type(outputs) is OrderedDict:  # output = (0 'out':?, 1 'aux':?)
                for index, (key, out) in enumerate(outputs.items()):
                    loss_record = criterion(out, masks)
                    if index == 0:#out
                        loss_record *= 0.5
                    else:#aux
                        loss_record *= 0.5
                    loss += loss_record
            else:
                loss = criterion(output, masks)


            # loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        current_lr = optimizer.param_groups[0]['lr']

        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {avg_train_loss:.4f} , lr: {current_lr}')
        # 調整學習率
        scheduler.step()
        # 每5個 epoch 後進行驗證
        if epoch % 1 == 0:
            model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    outputs = model(images)
                    if isinstance(outputs, OrderedDict):
                        preds = outputs['out'].max(1, keepdim=False)[1].cpu().numpy()
                    else:
                        preds = outputs.max(1, keepdim=False)[1].cpu().numpy()
                    all_preds.append(preds)
                    all_labels.append(masks.numpy())

            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            mean_iou = mean_iou_score(all_preds, all_labels)

            # 保存最佳模型
            if mean_iou > best_iou and epoch>=10:
                best_iou = mean_iou
                if not os.path.exists('checkpoint'):
                    os.makedirs('checkpoint')
                torch.save(model.state_dict(), f'checkpoint/DeepLabV3Model_{model_type}.pth')
                print(f"New best model saved with IoU: {best_iou:.4f}")

        # 在指定的 epoch 上進行圖像預測並保存結果
        if epoch in plot_epochs:
            model.eval()
            with torch.no_grad():
                for images  in test_plotset_loader:
                    images = images.to(device)
                    output = model(images)
                    if isinstance(output, OrderedDict):
                        pred = output['out'].max(1, keepdim=False)[1].cpu().numpy()
                    else:
                        pred = output.max(1, keepdim=False)[1].cpu().numpy()
# cls_color = {
#     0:  [0, 255, 255],
#     1:  [255, 255, 0],
#     2:  [255, 0, 255],
#     3:  [0, 255, 0],
#     4:  [0, 0, 255],
#     5:  [255, 255, 255],
#     6: [0, 0, 0],
# }
                    for i in range(3):
                        image = np.zeros((512, 512, 3), dtype=np.uint8)
                        image[pred[i] == 0] = [0, 255, 255]      # Cyan: Urban land
                        image[pred[i] == 1] = [255, 255, 0]      # Yellow: Agriculture land
                        image[pred[i] == 2] = [255, 0, 255]      # Purple: Rangeland
                        image[pred[i] == 3] = [0, 255, 0]        # Green: Forest land
                        image[pred[i] == 4] = [0, 0, 255]        # Blue: Water
                        image[pred[i] == 5] = [255, 255, 255]    # White: Barren land
                        image[pred[i] == 6] = [0, 0, 0]          # Black: Unknown
                        plot = Image.fromarray(image)
                        plot.save(os.path.join(plot_root, f'epoch-{epoch}-{i}.png'))


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
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate for training")
    # set_seed(9527)

    args = parser.parse_args()

    # 開始訓練
    train_model(num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
