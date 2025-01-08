import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from tokenizer import BPETokenizer

# Modified P2Dataset definition
class P2Dataset(Dataset):
    def __init__(self, image_root, json_path, mode='train', transform=None):
        self.image_root = image_root
        self.transform = transform
        self.mode = mode
        
        # Load JSON annotations
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        # Create a mapping from image IDs to image file names
        images = json_data["images"]
        annotations = json_data["annotations"]
        self.id2image = {image["id"]: os.path.join(image_root, image["file_name"]) for image in images}
        self.id2image2 = {image["id"]: image["file_name"] for image in images}

        # Create list of (image filename, caption) tuples
        self.filenames = [(self.id2image[anno["image_id"]], anno["caption"]) for anno in annotations]
        self.image_fn2= [self.id2image2[anno["image_id"]] for anno in annotations]
        # Initialize tokenizer if in train mode
        if self.mode == 'train':
            self.tokenizer = BPETokenizer(encoder_file='encoder.json', vocab_file='vocab.bpe')
        
        # Dataset length
        self.len = len(self.filenames)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Load image and caption (if in train mode)
        image_fn, caption = self.filenames[idx]
        image_fn2=self.image_fn2[idx].split('.')[0]
        image = Image.open(image_fn).convert("RGB")

        # Apply transformation if provided
        if self.transform:
            image = self.transform(image)
        
        if self.mode == 'train':
            # Encode the caption using BPE Tokenizer
            caption_tokens = self.tokenizer.encode(caption)
            caption_tokens.append(50256)  # Append end token

            # Prepare caption input and ground truth (gt)
            caption_input = caption_tokens.copy()
            caption_input.insert(0, 50256)  # Insert start token

            caption_input = torch.tensor(caption_input)
            caption_input = torch.nn.functional.pad(caption_input, (0, 50 - len(caption_input)), mode='constant', value=50256).long()

            gt = torch.tensor(caption_tokens, dtype=torch.int)
            gt = torch.nn.functional.pad(gt, (0, 50 - len(gt)), mode='constant', value=-100)
            
            return image, caption_input, gt
        else:
            # For validation mode, return image and filename
            return image, image_fn2

# Example usage
if __name__ == '__main__':
    transform =  transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((224, 224)),
                                    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
                                    )])
    
    dataset = P2Dataset(image_root='/home/remote/ylchang/dlcv/hw3/dlcv-fall-2023-hw3-aa30108962/hw3_data/p2_data/images/train/',
                        json_path='/home/remote/ylchang/dlcv/hw3/dlcv-fall-2023-hw3-aa30108962/hw3_data/p2_data/train.json',
                        mode='train',
                        transform=transform)
    
    print('# images in dataset:', len(dataset))  # Should print 60000
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1)
    
    data_iter = iter(dataloader)
    images, captions, gts = next(data_iter)
    
    print('Image tensor in each batch:', images.shape, images.dtype)
    print('Caption input tensor in each batch:', captions.shape, captions.dtype)
    print('Ground truth tensor in each batch:', gts.shape, gts.dtype)
