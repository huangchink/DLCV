import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from P3_dataloader import TextualInversionDataset, model_config
from ldm.util import instantiate_from_config
from torchvision import transforms

# 設置隨機種子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

# 定義模型載入函數
def load_model(ckpt_path, model_config, device):
    model = instantiate_from_config(model_config)
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading model weights from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully.")
    else:
        raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")
    return model.to(device)

# 定義訓練函數
def finetune_embedding(model, tokenizer, text_encoder, newtoken, init_token, data_root, device, attribute, num_epochs=100):
    # 設定新 token 的嵌入初始化
    tokenizer.add_tokens(newtoken)
    new_token_id = tokenizer.convert_tokens_to_ids(newtoken)
    init_token_id = tokenizer.encode(init_token, add_special_tokens=False)[0]
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeddings = text_encoder.get_input_embeddings().weight.data
    token_embeddings[new_token_id] = token_embeddings[init_token_id]
    
    # 定義 Dataset 和 Dataloader
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(p=0.2),  # Adds horizontal flip with 50% probability
        transforms.ToTensor(),
    ])

    # transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    dataset = TextualInversionDataset(root=data_root, tokenizer=tokenizer, newtoken=newtoken, attribute=attribute, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 設定優化器與學習率調度器
    embedding_parameter = text_encoder.get_input_embeddings().parameters()   

    optimizer = torch.optim.AdamW(embedding_parameter, lr=1e-02)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_loss = 0.1

    print(f'Training for {newtoken} started')
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, token_ids in dataloader:
            images = images.to(device)
            token_ids = token_ids.to(device)
            optimizer.zero_grad()
            
            # 獲取 conditioning 向量
            conditioning = text_encoder(input_ids=token_ids)[0]
            latent_images = model.encode_first_stage(images)
            latent_images = model.get_first_stage_encoding(latent_images).detach()
            noise = torch.randn_like(latent_images)
            timesteps = torch.randint(0, model.num_timesteps, (latent_images.shape[0],), device=device).long()
            noisy_latents = model.q_sample(latent_images, timesteps, noise=noise)
            noise_pred = model.apply_model(noisy_latents, timesteps, conditioning)
            
            # 計算損失
            loss = model.get_loss(pred=noise_pred, target=noise)
            loss.backward()

            # 凍结嵌入
            n_embed = len(text_encoder.get_input_embeddings().weight.grad)
            for i in range(n_embed - 1):
                text_encoder.get_input_embeddings().weight.grad[i] = 0
            
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
        scheduler.step()

        # 保存最佳模型權重
        # if avg_loss < best_loss:
        #     best_loss = avg_loss
        #     os.makedirs('embedding_checkpoint', exist_ok=True)
        #     torch.save(text_encoder.get_input_embeddings().state_dict(), f'./embedding_checkpoint/{newtoken}_embedding_{epoch}.pth')
        #     print(f"New best model saved for {newtoken} with loss {best_loss:.4f}")
        if epoch>=50:
            if avg_loss < best_loss:

                torch.save(text_encoder.get_input_embeddings().state_dict(), f'./embedding_checkpoint/{newtoken}_embedding_{epoch}.pth')
                print(f"New best model saved for {newtoken} with loss {best_loss:.4f}")
    print(f"Training for {newtoken} finished")


# 主函數
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "./ldm/models/stable-diffusion-v1/model.ckpt"
    model = load_model(ckpt_path, model_config, device)
    tokenizer = model.cond_stage_model.tokenizer
    text_encoder = model.cond_stage_model.transformer

    # 訓練 <new1> 的嵌入
    finetune_embedding(model=model, tokenizer=tokenizer, text_encoder=text_encoder, newtoken="<new1>", init_token="photo", data_root='../hw2_data/textual_inversion/0', 
                       device=device, attribute='object')

    # 訓練 <new2> 的嵌入
    finetune_embedding(model=model, tokenizer=tokenizer, text_encoder=text_encoder, newtoken="<new2>", init_token="painting", data_root='../hw2_data/textual_inversion/1', 
                        device=device, attribute='style')

if __name__ == "__main__":
    main()
