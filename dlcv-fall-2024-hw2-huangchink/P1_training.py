import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from P1_dataloader import get_dataloader
from P1_model import ContextUnet
from torchvision.utils import save_image
from torch.utils.data import DataLoader, TensorDataset
import timm
from digit_classifier import MnistmClassifier, evaluate_model, load_checkpoint
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from torchvision import transforms

# Define transforms for MNIST-M and SVHN
MNISTM_TRANSFORM = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

SVHN_TRANSFORM = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def load_generated_images(output_dir, dataset_type, num_samples=50):
    """從生成的圖片資料夾讀取圖片並使用相應的 transform"""
    images = []
    labels = []
    
    if dataset_type == "mnistm":
        transform = MNISTM_TRANSFORM
    elif dataset_type == "svhn":
        transform = SVHN_TRANSFORM
    else:
        raise ValueError("Unknown dataset type")

    for digit in range(10):
        for i in range(1, num_samples + 1):
            img_path = os.path.join(output_dir, f'{digit}_{i:03d}.png')
            img = Image.open(img_path)
            img = transform(img)  # Apply the dataset-specific transform
            images.append(img)
            labels.append(digit)
    
    return torch.stack(images), torch.tensor(labels)
# 設置固定的隨機種子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)



#update 一次運算完再取值可以加速很多
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
        sqrt_alpha_bar_t = self.sqrt_alpha_bar_t[t_steps][:, None, None, None] #維度擴展
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
        #print(context_no.shape)

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





# 將圖片生成和保存分離出來，方便評估
def save_generated_images(ddpm_model, num_samples=50):
    output_dir_mnistm = './P1_output/image/mnistm'
    output_dir_svhn = './P1_output/image/svhn'
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

def train(model, combined_loader, device):
    epochs = 50
    timesteps = 500
    lr = 1e-3
    model_save_dir = "P1_output/model/"

    os.makedirs(model_save_dir, exist_ok=True)

    ddpm = DDPM(model, betas=(1e-4, 0.02), timesteps=timesteps, drop_prob=0.1)
    ddpm.to(device)

    optimizer = torch.optim.Adam(ddpm.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    # Load classifiers for evaluation
    mnistm_classifier = MnistmClassifier().to(device)
    load_checkpoint('Classifier.pth', mnistm_classifier)
    mnistm_classifier.eval()

    svhn_classifier = timm.create_model('resnet34_svhn', pretrained=True, num_classes=10).to(device)
    svhn_classifier.eval()

    best_acc = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        ddpm.train()
        ave_loss = 0
        for (x, digit_condition, dataset_condition) in tqdm(combined_loader):
            optimizer.zero_grad()
            x = x.to(device)
            digit_condition = digit_condition.to(device)
            dataset_condition = dataset_condition.to(device)

            loss = ddpm(x, digit_condition, dataset_condition)
            loss.backward()
            optimizer.step()

            ave_loss += loss.item() / len(combined_loader)
        
        print(f"Average Loss is {ave_loss}")
        scheduler.step()

        # 在每個 epoch 結束後生成圖像並評估準確度
        save_generated_images(ddpm, num_samples=50)

        # 加載生成的圖像進行分類器評估
        mnistm_images, mnistm_labels = load_generated_images('./P1_output/image/mnistm', 'mnistm', num_samples=50)
        svhn_images, svhn_labels = load_generated_images('./P1_output/image/svhn', 'svhn', num_samples=50)

        mnistm_correct, mnistm_total = evaluate_model(mnistm_classifier, DataLoader(TensorDataset(mnistm_images, mnistm_labels), batch_size=10), device)
        svhn_correct, svhn_total = evaluate_model(svhn_classifier, DataLoader(TensorDataset(svhn_images, svhn_labels), batch_size=10), device)

        avg_acc = (mnistm_correct / mnistm_total + svhn_correct / svhn_total) / 2
        print(f'Epoch {epoch+1} - MNIST-M accuracy: {mnistm_correct/mnistm_total:.4f}, SVHN accuracy: {svhn_correct/svhn_total:.4f}, Avg accuracy: {avg_acc:.4f}')

        if avg_acc >= best_acc:
            best_acc = avg_acc
            print(f'Best model saved with accuracy {best_acc:.4f}')
            torch.save(ddpm.state_dict(), os.path.join(model_save_dir, f"bestmodel{epoch}.pth"))

if __name__ == '__main__':
    set_seed(42)  # 確保每次結果一致

    # Get datasets
    dataloader = get_dataloader(batch_size=256, root_dir='./hw2_data/digits')

    # Initialize the Context UNet model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device : ", device)
    model = ContextUnet(in_channels=3, height=28, width=28, n_feat=128, n_cfeat=12, n_downs=2).to(device)
    print('training啟動!!!!')
    # Train the model
    ddpm = DDPM(model, betas=(1e-4, 0.02), timesteps=500, drop_prob=0.1)
    ddpm.to(device)
    train(model, dataloader, device)
