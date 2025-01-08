import os
import argparse
import torch
from torchvision.utils import save_image
from UNet import UNet
from utils import beta_scheduler
import torch.nn as nn
import numpy as np

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 設置隨機種子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

set_seed(42)

# 定義DDIM參數
T = 1000  # T 的索引範圍應該為 0 到 999
ddim_steps = 50  # DDIM步數
betas = beta_scheduler(T).to(device)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), alphas_cumprod[:-1]])
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

# 定義DDIM的反向過程
def ddim_sample(model, x_i, ddim_steps, eta=0):
    ts = torch.linspace(T, 1, ddim_steps).long().to(device)
    for i in range(ddim_steps - 1):
        t2 = ts[i] - 1
        t1 = ts[i + 1] - 1
        epsilon = model(x_i, torch.tensor([t2], dtype=torch.float32).to(device))
        std = eta * torch.sqrt((1 - alphas_cumprod[t1]) / (1 - alphas_cumprod[t2])) * torch.sqrt(1 - (alphas_cumprod[t2] / alphas_cumprod[t1]))
        pred_x0 = sqrt_alphas_cumprod[t1] * (x_i - (sqrt_one_minus_alphas_cumprod[t2] * epsilon)) / sqrt_alphas_cumprod[t2]
        direction_xt = torch.sqrt(1 - alphas_cumprod[t1] - std**2) * epsilon
        if eta > 0:
            z = torch.randn_like(x_i).to(device)
            x_i = pred_x0 + direction_xt + std * z
        else:
            x_i = pred_x0 + direction_xt
    return x_i

if __name__ == "__main__":
    # 命令行參數
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_dir', type=str, required=True, help="Path to the directory of predefined noises")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the directory for saving generated images")
    parser.add_argument('--model_ckpt', type=str, required=True, help="Path to the pretrained model weight")
    args = parser.parse_args()

    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)

    # 設置 UNet 模型
    model = UNet().to(device)

    # 加載預訓練的 UNet 模型權重
    checkpoint = torch.load(args.model_ckpt, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 讀取並加載所有噪聲
    noise_files = sorted([f for f in os.listdir(args.noise_dir) if f.endswith('.pt')])

    # 生成圖像
    for i, noise_file in enumerate(noise_files[:10]):  # 只處理前 10 張噪聲
        with torch.no_grad():
            # 加載噪聲
            noise = torch.load(os.path.join(args.noise_dir, noise_file), map_location="cpu").to(device)

            # 使用DDIM進行逆擴散生成圖像
            generated_img = ddim_sample(model, noise, ddim_steps, eta=0)

            # Min-Max Normalization
            min_val = generated_img.min()
            max_val = generated_img.max()
            generated_img = (generated_img - min_val) / (max_val - min_val)

            # 保存生成的圖像
            output_path = os.path.join(args.output_dir, f'{i:02d}.png')
            save_image(generated_img, output_path)
            print(f'Image saved: {output_path}')
