# Install transformers >= 4.35.3
import os
import json
import torch
from transformers import pipeline, AutoProcessor
from PIL import Image
from torch.utils.data import DataLoader
from P1_dataloader import P1Dataset
import torchvision.transforms as transforms
from transformers import AutoProcessor, LlavaForConditionalGeneration
import argparse

# Argument parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, required=True, help="Path to the directory for input images")
parser.add_argument("--output_json", type=str, required=True, help="Path to the directory for json output")
args = parser.parse_args()


# Load model and processor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(device)
processor = AutoProcessor.from_pretrained(model_id)

# Define image root and dataloader
image_root = args.image_folder
dataset = P1Dataset(image_root=image_root, transform=transforms.Compose([
    transforms.ToTensor()
]))
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Initialize results dictionary
results = {}

# Iterate over images and perform inference
with torch.no_grad():
    for images, filenames in dataloader:
        for i in range(len(images)):
            image = images[i]  # Extract each image from the batch
            img_pil = transforms.ToPILImage()(image)  # Convert tensor to PIL image

            # Define the conversation for each image
            # conversation = [
            #                 {
            #                     "role": "user",
            #                     "content": [
            #                         {"type": "text", "text": "Provide a brief caption of the given image:"},
            #                         {"type": "image"},
            #                     ],
            #                 },
            #             ]

            # Apply chat template to get prompt
            # prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            prompt1 = "USER: <image>\nProvide a brief caption of the given image ASSISTANT:"
            prompt2 = "USER: <image>\nDescribe the image caption in a sentence ASSISTANT:"

            # Run inference
            inputs = processor(images=img_pil, text=prompt2, return_tensors='pt').to(device, torch.float16)
            outputs = model.generate(**inputs, max_new_tokens=30, num_beams=3, do_sample=False)
            caption = processor.decode(outputs[0], skip_special_tokens=True)

            # print(caption)
            # print(outputs[0]["generated_text"])
            caption = caption.split("ASSISTANT:")[-1].strip()
            #print(caption)
            # break
            # Store result without file extension
            filename_no_ext = os.path.splitext(filenames[i])[0]
            results[filename_no_ext] = caption

# Save results to JSON
output_file = args.output_json
output_dir = os.path.dirname(output_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_file}")
