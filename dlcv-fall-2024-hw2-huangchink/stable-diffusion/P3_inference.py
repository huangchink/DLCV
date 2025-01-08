import os
import json
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch
from ldm.util import instantiate_from_config
import numpy as np
from tqdm import tqdm
from P3_dataloader import model_config
import argparse
import gc
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def save_image(image_tensor, output_path):
    image = transforms.ToPILImage()(image_tensor)
    image.save(output_path)

@torch.no_grad()
def generate_and_save_images(model, sampler, json_path, output_dir, device):
    model.eval()

    # 讀取 input.json 來取得所有 prompts
    with open(json_path, 'r') as f:
        data_info = json.load(f)

    # 迭代每個 source 及其對應的 prompts
    for source_idx, entry in data_info.items():
        prompts = entry['prompt']  # 獲取當前 source 的 prompts 列表

        for prompt_idx, prompt in enumerate(prompts):
            print(f'Generating for prompt {prompt_idx} from source {source_idx}')
            prompt_output_dir = os.path.join(output_dir, str(source_idx), str(prompt_idx))
            os.makedirs(prompt_output_dir, exist_ok=True)

            # 獲取 prompt 的 conditioning (嵌入向量)
            text_embeddings = model.get_learned_conditioning([prompt]).to(device)
            shape = (model.channels, model.image_size, model.image_size)
            batch_size = 1

            for i in tqdm(range(25)):
                text_embeddings_batch = text_embeddings.repeat(batch_size, 1, 1)

                # 使用 DPMSolverSampler 進行采樣
                samples, _ = sampler.sample(
                    S=40,
                    conditioning=text_embeddings_batch,
                    batch_size=batch_size,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=10,
                    unconditional_conditioning=model.get_learned_conditioning([""] * batch_size)
                )

                # 解碼整個批量的樣本
                decoded_images = model.decode_first_stage(samples)
                # 將每張解碼的圖像轉換為 RGB 並儲存
                # for img_idx in range(batch_size):
                #     output_image = ((decoded_samples[img_idx] + 1.0) / 2.0).clamp(0.0, 1.0)
                #     output_image = transforms.ToPILImage()(output_image.cpu())
                #     output_path = os.path.join(prompt_output_dir, f'source{source_idx}_prompt{prompt_idx}_{i * batch_size + img_idx}.png')
                #     output_image.save(output_path)
                decoded_images  = ((decoded_images  + 1.0) / 2.0).clamp(0.0, 1.0)
                decoded_images  = decoded_images.cpu().numpy()[0].transpose(1, 2, 0) * 255
                output_image = Image.fromarray(decoded_images.astype(np.uint8))
                output_path = os.path.join(prompt_output_dir, f'source{source_idx}_prompt{prompt_idx}_{i}.png')
                output_image.save(output_path)
                # 釋放顯存
                del samples, decoded_images
                torch.cuda.empty_cache()
                gc.collect()

            print(f'Generated 25 images for source {source_idx}, prompt {prompt_idx}')
    print("All images generated and saved.")

def load_model(ckpt_path, model_config, device):
    model = instantiate_from_config(model_config)
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading model weights from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)

        print("Model loaded successfully.")
    else:
        raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")
    return model.to(device)

def load_custom_embedding(tokenizer, text_encoder, embed_path, new_token, device):
    tokenizer.add_tokens(new_token)
    text_encoder.resize_token_embeddings(len(tokenizer))
    learned_embeds_dict = torch.load(embed_path, map_location=device)
    placeholder_token_id = tokenizer.convert_tokens_to_ids(new_token)
    text_encoder.get_input_embeddings().weight.data[placeholder_token_id] = learned_embeds_dict['weight'][placeholder_token_id]
    print(f"Loaded embedding for token '{new_token}' from {embed_path}.")

def main():
    set_seed(42)
    print('starting inferencing')
    parser = argparse.ArgumentParser(description='P3 Inference with Custom Embedding')
    parser.add_argument('--json_path', type=str, required=True, help='Path to JSON file with prompts')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save generated images')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt_path, model_config, device)
    embed_path = 'new1best.pth'
    embed_path2 = 'new2best.pth'

    tokenizer = model.cond_stage_model.tokenizer
    text_encoder = model.cond_stage_model.transformer
    load_custom_embedding(tokenizer, text_encoder, embed_path, '<new1>', device)
    load_custom_embedding(tokenizer, text_encoder, embed_path2, '<new2>', device)

    # 初始化 DPMSolverSampler
    sampler = DPMSolverSampler(model)
    generate_and_save_images(model, sampler, args.json_path, args.output_dir, device)

if __name__ == "__main__":
    main()
