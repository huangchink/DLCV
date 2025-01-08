import torch
import os
from torchvision.utils import save_image
from UNet import UNet
import torch.nn as nn
import numpy as np
from utils import beta_scheduler
import torchvision.transforms as transforms

# 設置設備（使用 GPU 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
noise_dir = './hw2_data/face/noise/'  # 噪聲存放路徑
output_dir = './Output_folder/face'  # 輸出圖片存放路徑
unet_ckpt_path = './hw2_data/face/UNet.pt'  # 模型權重存放路徑

os.makedirs(output_dir, exist_ok=True)

# 設置 UNet 模型
model = UNet().to(device)

# 加載預訓練的 UNet 模型權重
checkpoint = torch.load(unet_ckpt_path, map_location=device)
model.load_state_dict(checkpoint)
print(f'Loaded UNet model weights from {unet_ckpt_path}')

model.eval()  # 設置模型為評估模式

# 定義DDIM的超參數
T = 1000  # T 的索引範圍應該為 0 到 999
ddim_steps = 50  # 任務中的DDIM步數
betas = beta_scheduler(T).to(device)  # 擴散過程中的beta值
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), alphas_cumprod[:-1]])

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

# 定義圖片轉換
transform = transforms.Compose([transforms.ToTensor()])

# Helper 函數來進行反向擴散（DDIM Sampling）
def ddim_sample(model, x_i, ddim_steps, eta=0):
    # 建立從 T+1=1000 到 1 的時間步數範圍，並均勻取 50 個步驟
    ts = torch.linspace(T, 1, ddim_steps).long().to(device)

    for i in range(ddim_steps - 1):
        # 在索引時將時間步數減去 1，以適應從 0 到 999 的索引範圍
        t2 = ts[i] - 1
        t1 = ts[i + 1] - 1

        # 根據DDIM邏輯，計算噪聲和每一步的生成過程
        epsilon = model(x_i, torch.tensor([t2], dtype=torch.float32).to(device))

        # 計算標準差
        std = eta * torch.sqrt((1 - alphas_cumprod[t1]) / (1 - alphas_cumprod[t2])) * torch.sqrt(1 - (alphas_cumprod[t2] / alphas_cumprod[t1]))

        # 計算第一項
        pred_x0 = sqrt_alphas_cumprod[t1] * (x_i - (sqrt_one_minus_alphas_cumprod[t2] * epsilon)) / sqrt_alphas_cumprod[t2]

        # 計算第二項
        direction_xt = torch.sqrt(1 - alphas_cumprod[t1] - std**2) * epsilon

        # 如果有噪聲，則加入隨機噪聲
        if eta > 0:
            z = torch.randn_like(x_i).to(device)
            x_i = pred_x0 + direction_xt + std * z
        else:
            x_i = pred_x0 + direction_xt

    return x_i

# 讀取並加載所有噪聲
noise_files = sorted([f for f in os.listdir(noise_dir) if f.endswith('.pt')])

# 設置不同的 eta 值
etas = [0.0, 0.25, 0.5, 0.75, 1.0]

grid = torch.Tensor().to(device)

with torch.no_grad(): 
    for i, eta in enumerate(etas):
        # 內層迴圈，生成每個 eta 對應的 4 張圖片
        for j, noise_file in enumerate(noise_files[:4]):  # 只用 00.pt ~ 03.pt
            # 加載噪聲
            noise = torch.load(os.path.join(noise_dir, noise_file), map_location="cpu").to(device)

            # 使用DDIM進行逆擴散生成圖像
            out = ddim_sample(model, noise, ddim_steps, eta)
            # 標準化圖像範圍從 [-1, 1] 到 [0, 1]
            out = (out + 1) / 2

            # 如果第一次執行，初始化 grid
            if i == 0 and j == 0:
                grid = out
            else:
                grid = torch.cat((grid, out), 0)  # 拼接生成的圖片

# 保存圖片網格
save_image(grid, './Output_folder/face/eta_compare.png', nrow=4, padding=2)

print("Image saved ")
