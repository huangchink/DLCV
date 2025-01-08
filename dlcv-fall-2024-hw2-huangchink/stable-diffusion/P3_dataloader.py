import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import random
import torch
from ldm.util import instantiate_from_config  # 載入這個工具來處理模型實例化
from torchvision import transforms



    # v1-inference 配置
model_config = {
        "target": "ldm.models.diffusion.ddpm.LatentDiffusion",
        "params": {
            "linear_start": 0.00085,
            "linear_end": 0.0120,
            "num_timesteps_cond": 1,
            "log_every_t": 200,
            "timesteps": 1000,
            "first_stage_key": "jpg",
            "cond_stage_key": "txt",
            "image_size": 64,
            "channels": 4,
            "cond_stage_trainable": False,
            "conditioning_key": "crossattn",
            "scale_factor": 0.18215,
            "use_ema": False,
            "unet_config": {
                "target": "ldm.modules.diffusionmodules.openaimodel.UNetModel",
                "params": {
                    "image_size": 32,
                    "in_channels": 4,
                    "out_channels": 4,
                    "model_channels": 320,
                    "attention_resolutions": [4, 2, 1],
                    "num_res_blocks": 2,
                    "channel_mult": [1, 2, 4, 4],
                    "num_heads": 8,
                    "use_spatial_transformer": True,
                    "transformer_depth": 1,
                    "context_dim": 768,
                    "use_checkpoint": True,
                    "legacy": False
                }
            },
            "first_stage_config": {
                "target": "ldm.models.autoencoder.AutoencoderKL",
                "params": {
                    "embed_dim": 4,
                    "monitor": "val/rec_loss",
                    "ddconfig": {
                        "double_z": True,
                        "z_channels": 4,
                        "resolution": 256,
                        "in_channels": 3,
                        "out_ch": 3,
                        "ch": 128,
                        "ch_mult": [1, 2, 4, 4],
                        "num_res_blocks": 2,
                        "attn_resolutions": [],
                        "dropout": 0.0
                    },
                    "lossconfig": {
                        "target": "torch.nn.Identity"
                    }
                }
            },
            "cond_stage_config": {
                "target": "ldm.modules.encoders.modules.FrozenCLIPEmbedder"
            }
        }
    }

class TextualInversionDataset(Dataset):
    def __init__(self,root,tokenizer,newtoken="<new1>",attribute = 'object',transform=None):

        self.root = root
        self.tokenizer = tokenizer
        self.attribute =attribute
        self.newtoken = newtoken
        self.transform = transform if transform else transforms.ToTensor()
        self.image_files = [f for f in os.listdir(self.root) if f.endswith('.jpg')]
        self.object_templates = [f"a photo of a {newtoken}",f"a photo of a small {newtoken}",f"a photo of a cute {newtoken}",f"a photo of a cool {newtoken}"]
        self.style_templates = [f"a photo in the style of {newtoken}",f"a painting in the style of {newtoken}",f"a nice painting in the style of {newtoken}",f"a beautiful painting in the style of {newtoken}"]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.image_files[idx])
        image = Image.open(image_path)
        image = self.transform(image)

        # if self.attribute == 'object':
        #     text = f"a photo of a {self.newtoken}"
        # elif self.attribute == 'style':
        #     text = f"a weird painting in the style of {self.newtoken}"
        if self.newtoken=="<new1>":
            text = random.choice(self.object_templates)
        else:
            text = random.choice(self.style_templates)

        tokenized = self.tokenizer(text, truncation=True, max_length=77, padding="max_length", return_tensors="pt")
        # print(text)
        return image, tokenized.input_ids[0]

# 定義模型載入函數
def load_model(ckpt_path, model_config, device):
    # 使用提供的配置來實例化模型
    model = instantiate_from_config(model_config)

    # 檢查是否提供了模型權重檔案路徑
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading model weights from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)

        # 提取 checkpoint 中的 state_dict
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint  # 如果沒有 state_dict，直接使用整個 checkpoint

        model.load_state_dict(state_dict, strict=False)  # 載入模型權重
        print("Model loaded successfully.")
    else:
        raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")

    return model.to(device)

if __name__ == "__main__":
    data_root  = '/home/remote/tchuang/DLCV/dlcv-fall-2024-hw2-huangchink/hw2_data/textual_inversion/1'
    ckpt_path = "./ldm/models/stable-diffusion-v1/model.ckpt"
    model = load_model(ckpt_path, model_config, 'cuda')

    # 在模型載入後調用該函數來生成 cond_ids
    # 初始化 TextualInversionDataset
    # tokenizer & text_encoder
    tokenizer = model.cond_stage_model.tokenizer
    text_encoder = model.cond_stage_model.transformer
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    # dataset = TextualInversionDataset(root=data_root,tokenizer=tokenizer,newtoken="<new1>",attribute='object',transform=transform)
    # # 測試資料集的 __getitem__ 方法
    # for i in range(min(5, len(dataset))):  # 測試前5個樣本
    #     image, ids = dataset[i]
    #     print(f"Example {i} - Image Shape: {image.shape}, Tokenized Input IDs: {ids}")  
    dataset = TextualInversionDataset(root=data_root,tokenizer=tokenizer,newtoken="<new2>",attribute='style',transform=transform)
    for i in range(min(5, len(dataset))):  # 測試前5個樣本
        image, ids = dataset[i]
        print(f"Example {i} - Image Shape: {image.shape}, Tokenized Input IDs: {ids}")  