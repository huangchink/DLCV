# Install transformers >= 4.35.3
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch.nn.functional as F

from transformers import BitsAndBytesConfig
import argparse
import cv2

# Argument parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, required=True, help="Path to the directory for input images")
parser.add_argument("--output_dir", type=str, required=True, help="Path to save attention visualizations")
args = parser.parse_args()

# Config for quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Load model and processor
model_id = "llava-hf/llava-1.5-7b-hf"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device)
model.config.use_cache = True  # Enable caching for generation
model.config.output_attentions = True  # Enable outputting attentions
model.config.return_dict_in_generate = True  # Return dict in generate

processor = AutoProcessor.from_pretrained(model_id)
tokenizer = processor.tokenizer  # Get the tokenizer

# Define image root and dataloader
image_root = args.image_folder
# dataset = P1Dataset(image_root=image_root, transform=transforms.Compose([
#     transforms.ToTensor()
# ]))
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Helper function to overlay attention map on the image
# def show_mask_on_image(image, mask):

#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
#     overlayed = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
#     return overlayed



with torch.no_grad():
    image_files = [f for f in os.listdir(args.image_folder) if f.endswith('.jpg')]
# Iterate over images and perform inference
    for filenames in image_files:
        image_path = os.path.join(args.image_folder, filenames)
        image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB format
        image_size = image.size  
        print(image_size)
        # Define the conversation for each image
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Provide a brief caption of the given image:"},
                    {"type": "image"},
                ],
            },
        ]

        # Apply chat template to get prompt
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors='pt').to(device, torch.float16)
        input_ids = inputs['input_ids']
        # print('input_ids',input_ids)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        # print('tokens',tokens)






        output = model.generate(**inputs, max_new_tokens=20,output_attentions=True)
        generated_ids = output.sequences[0]
        attentions = output.attentions  

        generated_text = processor.decode(generated_ids, skip_special_tokens=True)

        # Get attentions
        generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)
        caption = generated_text.split("ASSISTANT:")[-1].strip()


        generated_token_ids = generated_ids[600:]
        generated_tokens = tokenizer.convert_ids_to_tokens(generated_token_ids)
        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        words = generated_text.strip().split()
        token_idx = 0
        print(words)
        image = image.resize((336, 336))
        original_image_np = np.array(image) / 255.0  # Normalize to [0, 1] for overlay

        fig, axes = plt.subplots(nrows=(len(generated_tokens) // 5 + 1), ncols=5, figsize=(20, 10))
        axes = axes.flatten()
        ax = axes[0]
        ax.imshow(original_image_np)
        ax.set_title('<start>', fontsize=8)
        #imgae 是336x336 分 14 patch 每個patch size 是24x24 
        for i, token in enumerate(generated_tokens):

            attention_last_layer = attentions[i][0]  # first layer (1,32,1, 6xx)
            #print('last_layer_attns',attention_last_layer.shape)
            attn_32heads = attention_last_layer[0] # Shape: (32,1, 6xx)
            #print('attn_heads:',attn_32heads.shape)
            attn_32heads = attn_32heads[:,:, 5:581]

            # mean_attention = attn_32heads.mean(dim=0)  # 對所有 head 取平均(1, 576)


            head_index=0
            mean_attention = attn_32heads[head_index]  # 取特定某個head

            #print('mean_attention:',mean_attention.shape)
            mean_attention = mean_attention[0]

            attention_map = mean_attention.reshape(24, 24).detach().cpu().numpy()



            ax = axes[i+1]
            ax.imshow(original_image_np)
            ax.set_title(token, fontsize=8)

            # 取平均後的注意力權重

            attention_map = F.interpolate(torch.tensor(attention_map).unsqueeze(0).unsqueeze(0), size=(336, 336), mode="bilinear").squeeze().numpy()
            # attention_map = np.power(attention_map, 2)  # 增強對比度
            # attention_map = np.clip(attention_map, 0, 1)  # 對比度增強
            # attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

            # 疊加熱圖
            ax.imshow(attention_map, alpha=0.5, cmap="jet")
            ax.axis("off")

        # 隱藏多餘的子圖
        for j in range(len(words), len(axes)):
            axes[j].axis("off")

        # 保存圖像
        save_path = os.path.join(args.output_dir, f"ave_attention_{filenames}")
        print(f"attention visualizations saved to {save_path}")

        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.close()
print(f"Results and attention visualizations saved to {args.output_dir}")
