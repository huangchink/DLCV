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

def slerp(val, low, high):
    # 展平張量為 1D
    low_flat = low.view(-1)
    high_flat = high.view(-1)
    
    omega = torch.acos(torch.clip(torch.dot(low_flat / torch.norm(low_flat), high_flat / torch.norm(high_flat)), -1, 1))
    so = torch.sin(omega)
    
    if so == 0:
        return (1.0 - val) * low + val * high  # 線性插值
    
    slerp_result = torch.sin((1.0 - val) * omega) / so * low_flat + torch.sin(val * omega) / so * high_flat
    
    # 將結果重塑回原來形狀
    return slerp_result.view_as(low)


# 讀取並加載噪聲
noise_0 = torch.load(os.path.join(noise_dir, '00.pt'), map_location="cpu").to(device)
noise_1 = torch.load(os.path.join(noise_dir, '01.pt'), map_location="cpu").to(device)

alphas = torch.linspace(0, 1, 11).to(device)  # α 值範圍
with torch.no_grad():
    # 生成 Slerp 插值圖片
    slerp_images = []
    for alpha in alphas:
        interpolated_noise = slerp(alpha, noise_0, noise_1)
        out = ddim_sample(model, interpolated_noise, ddim_steps, eta=0)
        out = (out + 1) / 2  # 標準化範圍到 [0, 1]
        slerp_images.append(out)

    # 保存 Slerp 插值圖片網格
    slerp_grid = torch.cat(slerp_images, dim=0)
    save_image(slerp_grid, './Output_folder/face/slerp_interpolation.png', nrow=11, padding=2)

    # 生成線性插值圖片
    linear_images = []
    for alpha in alphas:
        interpolated_noise = (1 - alpha) * noise_0 + alpha * noise_1  # 線性插值
        out = ddim_sample(model, interpolated_noise, ddim_steps, eta=0)
        out = (out + 1) / 2  # 標準化範圍到 [0, 1]
        linear_images.append(out)

    # 保存線性插值圖片網格
    linear_grid = torch.cat(linear_images, dim=0)
    save_image(linear_grid, './Output_folder/face/linear_interpolation.png', nrow=11, padding=2)

print("Images for Slerp and Linear interpolation saved!")

