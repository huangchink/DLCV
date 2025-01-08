import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import json
import re
from tokenizer import BPETokenizer
import timm
from P2_dataloader import P2Dataset
import torch.optim as optim
from evaluate import *
import loralib as lora
import collections
from torch import Tensor
import math

class Config:
    def __init__(self, checkpoint=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.ln_3 = nn.LayerNorm(cfg.n_embd)

        self.attn = Attention(cfg)
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', lora.Linear(cfg.n_embd, 4 * cfg.n_embd,r=16)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', lora.Linear(4 * cfg.n_embd, cfg.n_embd,r=16))
        ]))
        # self.mlp2 = nn.Sequential(
        #     nn.Linear(cfg.n_embd, 1/4 * cfg.n_embd),
        #     nn.GELU(),
        #     nn.Linear(1/4 * cfg.n_embd, cfg.n_embd)
        # )
        self.atten = None

    def forward(self, x):
        temp,atten=self.attn(self.ln_1(x))

        x = x + temp
        x = x + self.mlp(self.ln_2(x))
        # x = x + self.mlp2(self.ln_3(x))

        return x,atten
class ModifiedDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            h=nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = lora.Linear(cfg.n_embd, cfg.vocab_size, bias=False,r=16)
        self.transformer.wte.weight = self.lm_head.weight
        self.ln = nn.LayerNorm(cfg.n_embd)

        # Projection layer for visual features
        self.visual_projection = nn.Linear(1408, 1088)

        self.visual_projection2 = nn.Linear(1088, cfg.n_embd)

        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)
            print('pretrain decoder loaded')


    def forward(self, x: Tensor, image_features: Tensor = None):
        if image_features is not None:
            visual_embed = self.visual_projection(image_features.float())
            visual_embed = self.visual_projection2(visual_embed)

            x = self.transformer.wte(x)
            visual_embed=self.ln(visual_embed)
            x = torch.cat((visual_embed, x), dim=1)

        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
        pos = torch.arange(x.size(1), dtype=torch.long, device=x.device).unsqueeze(0)
        x = x + self.transformer.wpe(pos)
        attn_list=[]
        for idx,block in enumerate(self.transformer.h):
            
            x ,atten= block(x)            
            attn_list.append(atten)
        # x = self.transformer.h(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits,attn_list
class Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x):
        B, T, C = x.size() # batch, context, embedding
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att1 = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att1.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C)),att1

class P3Dataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        初始化 P3Dataset。
        
        Parameters:
            image_dir (str): 圖片資料夾的路徑。
            transform (callable, optional): 圖片的轉換操作。
        """
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        根據索引獲取圖片及其文件名。

        Parameters:
            idx (int): 索引值。

        Returns:
            image (PIL.Image or Tensor): 圖片（如果有 transform，則為 Tensor）。
            filename (str): 圖片的文件名。
        """
        filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert("RGB")  # 確保圖片是 RGB 格式

        if self.transform:
            image = self.transform(image)

        return image, filename
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = timm.create_model('vit_giant_patch14_clip_224.laion2b', pretrained=True)

    def forward(self, x):
        out = self.model.forward_features(x)
        out = out[:, 1:, :]
        return out

class Model(nn.Module):

    def __init__(self,  decoder_checkpoint='./hw3_data/p2_data/decoder_model.bin'):
        super().__init__()
        # Create decoder configuration and model
        self.cfg = Config(decoder_checkpoint)
        self.decoder  = ModifiedDecoder(self.cfg).to(device)
        self.encoder = Encoder().to(device)

        data_config=timm.data.resolve_model_data_config(self.encoder)
        self.transforms_train=timm.data.create_transform(**data_config,is_training=True)
        self.transforms_test=timm.data.create_transform(**data_config,is_training=False)


    def forward(self, image, caption_input):
        encoder_out = self.encoder(image)
        # print('encoder_out.shape',encoder_out.shape)
        output,attn_list = self.decoder(caption_input, encoder_out)

        return output,attn_list
    
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F

if __name__ == "__main__":
    # 定義圖片的轉換操作
    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(([0.4850, 0.4560, 0.4060]), ([0.2290, 0.2240, 0.2250]))
    ])

    # 載入資料集
    image_dir = "/home/remote/tchuang/DLCV/dlcv-fall-2024-hw3-huangchink/hw3_data/p3_data/images/"
    dataset = P3Dataset(image_dir=image_dir, transform=transforms_test)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print('dataloader', len(dataloader))

    # 設置裝置
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化模型
    model = Model()
    model.to(device)

    # 加載指定的權重檔案
    checkpoint_path = "./checkpoint/best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    print(f"Loaded checkpoint from {checkpoint_path}")

    # 設置輸出資料夾
    output_dir = "./output_p3"
    os.makedirs(output_dir, exist_ok=True)

    # 推論並生成格式化的注意力熱圖序列
    model.eval()
    tokenizer = BPETokenizer(encoder_file='encoder.json', vocab_file='vocab.bpe')
    end_token = tokenizer.encoder.get("<|endoftext|>")

    print('predicting:')
    with torch.no_grad():
        for idx, (image, filename) in enumerate(dataloader):
            filename = filename[0]
            image = image.to(device)
            encoder_out = model.encoder(image)  # 提取圖像特徵
            original_image = Image.open(os.path.join(image_dir, filename)).resize((224, 224))  # Resize to 224x224

            # 初始化生成的序列
            current_caption = torch.full((1, 50), end_token, dtype=torch.int64, device=device)  # (1, 50)
            words = ["<start>"]  # 開始符號

            # 生成完整文字序列
            for current_idx in range(49):  # 最大生成長度為 49
                output, attn_list = model.decoder(current_caption, encoder_out)  # 解碼器前向傳播
                next_token = output.max(2, keepdim=False)[1][0, 256 + current_idx].item()

                # 若遇到終止 token，結束生成
                if next_token == end_token:
                    words.append(tokenizer.decode([next_token]))
                    break

                # 更新當前序列
                current_caption[0, current_idx + 1] = next_token
                words.append(tokenizer.decode([next_token]))

            print(words)

            # 只取第一層的注意力權重
            attention_last_layer = attn_list[1]  # 因為 最後一層效果不太好qq
            mean_attention = attention_last_layer.mean(dim=1)  # 對所有 head 取平均 (B, T, T)

            # 可視化注意力圖
            fig, axes = plt.subplots(nrows=(len(words) // 5 + 1), ncols=5, figsize=(20, 10))
            axes = axes.flatten()
            ax = axes[0]
            # Convert original image to numpy for plotting (resize to 224x224 if needed)
            original_image = original_image.resize((224, 224))
            original_image_np = np.array(original_image) / 255.0  # Normalize to [0, 1] for overlay

            ax.imshow(original_image_np)
            ax.set_title('<start>', fontsize=8)
            for i, word in enumerate(words):
                if i == 0:
                    continue
                ax = axes[i]
                ax.imshow(original_image_np)
                ax.set_title(word, fontsize=8)

                # 取平均後的注意力權重
                attention_weights = mean_attention[0, 255 + i, :256]  # 平均後的注意力權重

                # 重塑成影像大小
                attention_map = attention_weights.view(16, 16).cpu().numpy()
                attention_map = F.interpolate(torch.tensor(attention_map).unsqueeze(0).unsqueeze(0), size=(224, 224), mode="bilinear").squeeze().numpy()
                attention_map = np.power(attention_map, 2)  # 增強對比度
                attention_map = np.clip(attention_map, 0, 1)  # 對比度增強
                # attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

                # 疊加熱圖
                ax.imshow(attention_map, alpha=0.3, cmap="jet")
                ax.axis("off")

            # 隱藏多餘的子圖
            for j in range(len(words), len(axes)):
                axes[j].axis("off")

            # 保存圖像
            save_path = os.path.join(output_dir, f"ave_attention_{filename}")
            plt.tight_layout()
  
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
            plt.close()

    print(f"All attention sequences have been saved to {output_dir}")
