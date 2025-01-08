import os
import torch
import torch.nn as nn  # 引入 torch.nn 模組
from P1_model import ContextUnet  # 引入 UNet 模型
from torchvision.utils import save_image, make_grid
from PIL import Image
import numpy as np

# 設置固定的隨機種子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

set_seed(42)  # 使用42上帝的數字 作為固定的seed 

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
        self.model = model.to(device)
        param = ddpm_param(betas[0], betas[1], timesteps)
        for k, v in param.items():
            self.register_buffer(k, v)
        self.timesteps = timesteps 
        self.drop_prob = drop_prob

    def forward(self, x, digit_condition, dataset_condition):
        t_steps = torch.randint(1, self.timesteps + 1, (x.size(0),)).to(device)
        noise = torch.randn_like(x)
        sqrt_alpha_bar_t = self.sqrt_alpha_bar_t[t_steps][:, None, None, None]  # 維度擴展
        sqrt_1_sub_alpha = self.sqrt_1_sub_alpha[t_steps][:, None, None, None]
        x_t = sqrt_alpha_bar_t * x + sqrt_1_sub_alpha * noise

        digit_cond_mask = torch.bernoulli(torch.ones_like(digit_condition[:, :1]) * self.drop_prob).to(device)
        dataset_cond_mask = torch.bernoulli(torch.ones_like(dataset_condition[:, :1]) * self.drop_prob).to(device)

        digit_cond_mask = digit_cond_mask.repeat(1, digit_condition.size(1))
        dataset_cond_mask = dataset_cond_mask.repeat(1, dataset_condition.size(1))

        digit_condition = digit_condition * (1 - digit_cond_mask)
        dataset_condition = dataset_condition * (1 - dataset_cond_mask)

        context = torch.cat([digit_condition, dataset_condition], dim=1)
        t = t_steps / self.timesteps
        t = t[:, None]

        predict_noise = self.model(x_t, t, context)
        loss = nn.MSELoss()(noise, predict_noise)
        return loss

    def sample(self, n_sample, size, w=2, digit_condition=None, dataset_condition=None):
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
        context = context.repeat(n_sample, 1)
        context_no = torch.zeros_like(context).to(device)
        context_repeat = torch.cat([context, context_no], dim=0)

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

# 修改後的直接保存圖片的函數
def save_generated_images(images, save_path):
    """Save a grid of generated images."""
    grid_img = make_grid(images, nrow=10)
    save_image(grid_img, save_path)

@torch.no_grad()
def reverse_process_visualization(ddpm_model, digit_condition, dataset_condition, device, save_path_prefix, n_sample=10, timesteps=[1, 50, 100, 200, 300, 500]):
    """Visualize the reverse process of diffusion with different timesteps."""
    x_i = torch.randn(n_sample, 3, 28, 28).to(device)  # Start with random noise for n_sample images
    digit_condition = torch.tensor([digit_condition]).to(device)
    dataset_condition = torch.tensor([dataset_condition]).to(device)

    # One-hot encode the conditions
    digit_onehot = torch.nn.functional.one_hot(digit_condition, num_classes=10).float().to(device)
    dataset_onehot = torch.nn.functional.one_hot(dataset_condition, num_classes=2).float().to(device)
    context = torch.cat([digit_onehot, dataset_onehot], dim=1)

    context = context.repeat(n_sample, 1)
    context_no = torch.zeros_like(context).to(device)

    context_rep = torch.cat([context, context_no], dim=0)

    steps = []  # To store images at specific timesteps

    for i in range(ddpm_model.timesteps, 0, -1):
        x_i_rep = x_i.repeat(2, 1, 1, 1)
        t_s = torch.full((n_sample, 1), i / ddpm_model.timesteps, device=device)
        t_s_rep = t_s.repeat(2, 1)

        z = torch.randn_like(x_i_rep) if i > 1 else torch.zeros_like(x_i_rep)

        predict_out = ddpm_model.model(x_i_rep, t_s_rep, context_rep)
        conditional_out = predict_out[:n_sample]
        unconditional_out = predict_out[n_sample:]

        Merge_out = (1 + 2.0) * conditional_out - 2.0 * unconditional_out
        x_i = x_i[:n_sample]

        x_i = ddpm_model.one_over_sqrt_alpha_t[i] * (x_i - Merge_out * ddpm_model.complexterm[i]) + ddpm_model.sigma_t[i] * z[:n_sample]

        if i in timesteps:
            steps.append(x_i[1].clone().detach().cpu())  # Save only the second image
    if dataset_condition==0:
        save_image(torch.stack(steps), save_path_prefix+'/mnistm_'+f"reverse_process.png", nrow=len(timesteps))
    elif dataset_condition==1:
        save_image(torch.stack(steps), save_path_prefix+'/svhn_'+f"reverse_process.png", nrow=len(timesteps))

    torch.cuda.empty_cache()

@torch.no_grad()
def generate_images_grid(ddpm_model, device, output_dir="./P1_output/reportimgs"):
    """Generate grid images for MNIST-M and SVHN."""
    os.makedirs(output_dir, exist_ok=True)

    for dataset_type, dataset_cond in [("mnistm", 0), ("svhn", 1)]:
        all_images = []
        for digit in range(10):
            digit_condition = torch.tensor([digit]).to(device)
            dataset_condition_tensor = torch.tensor([dataset_cond]).to(device)
            images = ddpm_model.sample(n_sample=10, size=(3, 28, 28), w=2.0, digit_condition=digit_condition, dataset_condition=dataset_condition_tensor)
            all_images.append(images.cpu())

        all_images = torch.cat(all_images)
        save_path = os.path.join(output_dir, f"{dataset_type}_generated_grid.png")
        save_generated_images(all_images, save_path)

        print(f"Generated image grid for {dataset_type.upper()} saved at {save_path}")

if __name__ == "__main__":
    # Set up device and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContextUnet(in_channels=3, height=28, width=28, n_feat=128, n_cfeat=12, n_downs=2).to(device)

    ddpm = DDPM(model, betas=(1e-4, 0.02), timesteps=500, drop_prob=0.1)
    ddpm.load_state_dict(torch.load("./P1_output/model/bestDDPM.pth", map_location=device))

    # Step 2: Generate grid of images for MNIST-M and SVHN
    generate_images_grid(ddpm, device, output_dir="./P1_output/reportimgs")

    # Step 3: Visualize the reverse process for digit "0" in MNIST-M and SVHN
    reverse_process_visualization(ddpm, digit_condition=0, dataset_condition=0, device=device, save_path_prefix="./P1_output/reportimgs")
    reverse_process_visualization(ddpm, digit_condition=0, dataset_condition=1, device=device, save_path_prefix="./P1_output/reportimgs")
