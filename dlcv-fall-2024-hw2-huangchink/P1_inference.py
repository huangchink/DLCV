import os
import argparse
import torch
import torch.nn as nn
from P1_model import ContextUnet  # 假設模型定義在P1_model中
from torchvision.utils import save_image
import numpy as np

# 設置固定的隨機種子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


# 直接從 P1_training.py 使用 ddpm_param 和 DDPM 類
def ddpm_param(beta1, beta2, T):
    beta_t = beta1 + (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T 
    alpha_t = 1 - beta_t
    alpha_bar_t = torch.exp(torch.cumsum(torch.log(alpha_t), dim=0))
    return {
        "one_over_sqrt_alpha_t": 1 / torch.sqrt(alpha_t),
        "sigma_t": torch.sqrt(beta_t),
        "sqrt_alpha_bar_t": torch.sqrt(alpha_bar_t),
        "sqrt_1_sub_alpha": torch.sqrt(1 - alpha_bar_t),
        "complexterm": (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t),
    }

class DDPM(nn.Module):
    def __init__(self, model, betas, timesteps, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.model = model
        param = ddpm_param(betas[0], betas[1], timesteps)
        for k, v in param.items():
            self.register_buffer(k, v)
        self.timesteps = timesteps
        self.drop_prob = drop_prob

    def sample(self, n_sample, size, w=0.0, digit_condition=None, dataset_condition=None):
            xt = torch.randn(n_sample, *size).to(device)
            if digit_condition is not None and dataset_condition is not None:
                digit_labels = digit_condition
                dataset_labels = dataset_condition
            else:
                digit_labels = torch.randint(0, 10, (n_sample,), device=device)
                dataset_labels = torch.randint(0, 2, (n_sample,), device=device)

            digit_condition = torch.nn.functional.one_hot(digit_labels, num_classes=10).float().to(device)
            dataset_condition = torch.nn.functional.one_hot(dataset_labels, num_classes=2).float().to(device)
            
            

            context = torch.cat([digit_condition, dataset_condition], dim=1)
            context=context.repeat(n_sample, 1)
            # print(context.shape)
            context_no = torch.zeros_like(context).to(device)

            # print(c.shape)
            context_repeat = torch.cat([context, context_no], dim=0)  # 確保  [2*n_sample, 12]


            for i in range(self.timesteps, 0, -1):
                xt_repeat = xt.repeat(2, 1, 1, 1)

                t_s = torch.full((n_sample, 1), i / self.timesteps, device=device)
                t_repeat = t_s.repeat(2, 1)  

                z = torch.randn_like(xt_repeat) if i > 1 else torch.zeros_like(xt_repeat)

                predict_noise = self.model(xt_repeat, t_repeat, context_repeat)
                conditional_out = predict_noise[:n_sample]
                unconditional_out = predict_noise[n_sample:]
                ep = (1 + w) * conditional_out - w * unconditional_out
                xt = xt_repeat[:n_sample]

                xt = self.one_over_sqrt_alpha_t[i] * (xt - ep * self.complexterm[i]) + self.sigma_t[i] * z[:n_sample]

            return xt

@torch.no_grad()
# 將圖片生成和保存分離出來，方便評估
def save_generated_images(ddpm_model, output_dir,num_samples=50):
    output_dir_mnistm =os.path.join(output_dir,'mnistm')
    output_dir_svhn = os.path.join(output_dir,'svhn')
    os.makedirs(output_dir_mnistm, exist_ok=True)
    os.makedirs(output_dir_svhn, exist_ok=True)

    ddpm_model.eval()
    with torch.no_grad():
        for digit in range(10):
            mnistm_img = ddpm_model.sample(n_sample=num_samples, size=(3, 28, 28),  w=2, digit_condition=torch.tensor([digit]).to(device), dataset_condition=torch.tensor([0]).to(device))
            svhn_img = ddpm_model.sample(n_sample=num_samples, size=(3, 28, 28),  w=2, digit_condition=torch.tensor([digit]).to(device), dataset_condition=torch.tensor([1]).to(device))

            for i in range(num_samples):
                mnistm_img_path = os.path.join(output_dir_mnistm, f'{digit}_{i+1:03d}.png')
                save_image(mnistm_img[i], mnistm_img_path)
                svhn_img_path = os.path.join(output_dir_svhn, f'{digit}_{i+1:03d}.png')
                save_image(svhn_img[i], svhn_img_path)

if __name__ == "__main__":
    # 解析命令行參數
    set_seed(42)  # 設定固定的seed

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory for saving generated images")
    args = parser.parse_args()

    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加載模型
    model = ContextUnet(in_channels=3, height=28, width=28, n_feat=128, n_cfeat=12, n_downs=2).to(device)
    ddpm = DDPM(model, betas=(1e-4, 0.02), timesteps=500, drop_prob=0.1)
    ddpm.load_state_dict(torch.load("bestDDPM.pth", map_location=device))
    

    # 生成並保存圖片
    save_generated_images(ddpm, args.output_dir, num_samples=50)
