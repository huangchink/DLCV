import os
import clip
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import json
import re
from tokenizer import BPETokenizer
import timm
from P2_dataloader import P2Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from evaluate import *
from torch import Tensor
from tqdm import tqdm
import loralib as lora
from torch.cuda.amp import autocast
import collections
import math

# Decoder class definition
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
        temp=self.attn(self.ln_1(x))
        self.atten = self.attn.atten

        x = x + temp
        x = x + self.mlp(self.ln_2(x))
        # x = x + self.mlp2(self.ln_3(x))

        return x
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
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
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
            
            x = block(x)            
            attn_list.append(block.atten)
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
        self.atten = None

    def forward(self, x):
        B, T, C = x.size() # batch, context, embedding
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        self.atten = att

        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))

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

# Main script
if __name__ == "__main__":
    image_root = "./hw3_data/p2_data/images/val"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model=Model()
    # Load CLIP model
    # Initialize Tokenizer
    # transforms_train = model.transforms_train
    # transforms_test = model.transforms_test
    # Define transformations
    transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.2),
        # transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4), hue=None),
        transforms.ToTensor(),
        transforms.Normalize(([0.4850, 0.4560, 0.4060]), ([0.2290, 0.2240, 0.2250]))
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(([0.4850, 0.4560, 0.4060]), ([0.2290, 0.2240, 0.2250]))
    ])
    # print(transforms_train)
    model = model.to(device)
    # Define transformations

    # Create training DataLoader
    train_dataset = P2Dataset(image_root='./hw3_data/p2_data/images/train', 
                              json_path='./hw3_data/p2_data/train.json', 
                              mode='train', transform=transforms_train)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

    # Create validation DataLoader
    val_dataset = P2Dataset(image_root=image_root, 
                            json_path='./hw3_data/p2_data/val.json', 
                            mode='val', transform=transforms_test)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)



    lora.mark_only_lora_as_trainable(model)
    model.decoder.visual_projection.requires_grad_(True)
    model.decoder.visual_projection2.requires_grad_(True)

    tokenizer = BPETokenizer(encoder_file='encoder.json', vocab_file='vocab.bpe')
    # print(f"## Model #param={sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6}M") 
    # for layer in model.decoder.transformer.h: 
    #     for name, param in layer.cross_attn.named_parameters():
    #         param.requires_grad = True 
    #     for name, param in layer.ln_3.named_parameters():
    #         param.requires_grad = True 
    # for name, param in model.encoder.named_parameters():
    #     param.requires_grad = False
    print(f"Total params : {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6}M") 

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # Set end token manually
    end_token = tokenizer.encoder.get("<|endoftext|>")
    print('end_token:',end_token)
    # optimizer = optim.AdamW(trainable_params, lr=1e-5)
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)

    epochs =10
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9, last_epoch=-1, verbose='deprecated')
    # scheduler = optim.lr_scheduler.StepLR(
    #                 optimizer, 
    #                 step_size=2, 
    #                 gamma=0.1
    #             )
    best_clip_score = -float('inf')
    print('training 啟動!')
    for epoch in tqdm(range(epochs)):
        # Training step
        model.train()
        total_loss = 0
        batch_count = 0
        for image, caption_input, gt in train_dataloader:
            optimizer.zero_grad()
            image = image.to(device)
            caption_input = caption_input.to(device)
            gt = gt.to(device).to(torch.int64)  

            with autocast():
                output,attn_list = model(image,caption_input)
        
                # 調整 output 的大小以符合 gt
                output = output[:, output.size(1)-gt.size(1):, :]

                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(output.reshape(-1, model.cfg.vocab_size), gt.reshape(-1))


                # 累加總損失
                total_loss += loss.item()
                batch_count += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.2)

            # torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.2)
            optimizer.step()
        scheduler.step()

        # 計算並打印該 epoch 的平均損失
        avg_loss = total_loss / batch_count
        print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_loss:.4f}")
        # 每個 epoch 完成後進行評估


        if epoch<5:
            continue
        model.eval()
        predictions = {}
        print('predicting:')
        with torch.no_grad():
            for idx, (image, filename) in enumerate(val_dataloader):
                image = image.to(device)
                encoder_out = model.encoder(image)  # 提取圖像特徵
                
                # 初始化生成的序列，並設置長度 50，以 50256 作為起始和填充 token
                current_caption = torch.full((1, 50), end_token, dtype=torch.int64, device=device)  # (1, 50)
                
                word = ""  # 用於存儲生成的文本
                for current_idx in range(49):  # 生成最多 49 個 token
                    # 解碼器前向傳播
                    output,attn_list = model.decoder(current_caption, encoder_out)  # output shape: (1, 50, vocab_size)
                    
                    # 在當前時間步的最後一個位置上獲取最可能的 token
                    next_token = output.max(2, keepdim=False)[1][0, output.size(1)-gt.size(1)+current_idx].item()

                    # 終止條件：遇到終止 token 或達到最大生成長度
                    if next_token == end_token:
                        break
                    
                    
                    # 更新 `caption_test` 中的下一個位置的 token
                    current_caption[0, current_idx + 1] = next_token
                    
                    # 將當前 token 解碼成文字，並拼接到 `word`
                    word += tokenizer.decode([next_token])

                # 儲存生成的結果
                predictions[filename[0]] = word
                

                # print(f"word:{word}")

        # 保存生成的結果到 JSON 文件
        output_path = "predB.json"
        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=4)


        # 計算 CLIP 分數
        annotations = readJSON('./hw3_data/p2_data/val.json')

        # Preprocess annotation file
        gts = getGTCaptions(annotations)
            # 查看 `gts` 和 `predictions` 的 keys 差異
        pred_keys = set(predictions.keys())
        gts_keys = set(gts.keys())

        print("Keys in predictions but not in gts:", pred_keys - gts_keys)
        print("Keys in gts but not in predictions:", gts_keys - pred_keys)

        # 確認 keys 是否一致
        assert pred_keys == gts_keys, "Mismatch between predictions and ground truth keys."

        # Check predictions content is correct
        assert type(predictions) is dict
        assert set(predictions.keys()) == set(gts.keys())
        assert all([type(pred) is str for pred in predictions.values()])

        # CIDErScore
        cider_score = CIDERScore()(predictions, gts)

        # CLIPScore
        clip_score = CLIPScore()(predictions, image_root)
        
        # print(f"CIDEr: {cider_score}")
        print(f"Epoch [{epoch + 1}/{epochs}]")
        print(f"CIDEr: {cider_score} | CLIPScore: {clip_score}")
        best_model_path = f"./checkpoint/best_model{epoch}.pth"

        trainable_param_names = [name for name, param in model.named_parameters() if param.requires_grad]
        save_weight = {k: v for k, v in model.state_dict().items() if k in trainable_param_names}

        # print(len(save_weight))  # 應該不會再是 0
        # 保存具有最佳 CLIP 分數的模型
        if clip_score >= 0.73 and cider_score>=0.94:
            best_clip_score = clip_score
            # save_weight = { k:v for k,v in model.state_dict().items() if k in trainable_params}
            # print(len(save_weight))
            torch.save(save_weight, best_model_path)
            print(f"New best model saved with CLIP Score: {clip_score:.4f}")
