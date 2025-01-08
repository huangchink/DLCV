import torch
from torch.utils.data import DataLoader
from torchvision import models
from byol_pytorch import BYOL
from P1_dataloader import MiniImageNetDataset, TRANSFORM_IMG  # 從 P1_dataloader 引入
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import CosineAnnealingLR

def BYOL_resnet_pretraining(image_dir):
    # 選擇 device，若有 GPU 則使用 GPU，否則使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 使用 MiniImageNetDataset
    dataset = MiniImageNetDataset(image_dir, transform=TRANSFORM_IMG)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

    # 初始化 ResNet50 模型並移動到選定的 device
    resnet = models.resnet50(weights=None)
    resnet = resnet.to(device)
    
    # 使用 BYOL 進行 self-supervised learning 訓練
    learner = BYOL(
        resnet,
        image_size=128,  # 根據你的資料集調整為 128 x 128
        hidden_layer='avgpool'
    )
    
    # 將 BYOL 移動到選定的 device
    learner = learner.to(device)
    
    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500, eta_min=0)
   
    best_loss =9999.0
    save_path = './checkpoint/SSL-Resnet.pt'
    
    # 進行 100 個 epoch 的 self-supervised 訓練
    for epoch in tqdm(range(500)): 
        print(f'Epoch: {epoch+1}')
        epoch_loss = 0

        for images in dataloader:
            images = images.to(device)  # 將圖片移動到選定的 device
            loss = learner(images)
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average()  # 更新 target encoder
            epoch_loss += loss.item()

        # 計算平均損失
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/100] - Avg Loss: {avg_loss:.4f}")

        # 如果當前平均損失小於最小損失，則保存模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(resnet.state_dict(), save_path)
            print(f"Model saved with loss: {best_loss:.4f}")
        scheduler.step()  # 更新學習率

if __name__ == "__main__":
    print("pretraining start!")

    image_dir ='../hw1_data/p1_data/mini/train'
    BYOL_resnet_pretraining(image_dir)
