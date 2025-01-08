import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import random

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mode='train', transform=None, augment=True):
        self.img_dir = img_dir
        self.mode = mode
        self.transform = transform
        self.augment = augment
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith('_sat.jpg')])

    def __len__(self):
        return len(self.images)

    def random_flip(self, image, mask):
        if random.random() >= 0.5:
            image = image[:, ::-1, :].copy()
            mask = mask[:, ::-1].copy()
        if random.random() >= 0.5:
            image = image[::-1, :, :].copy()
            mask = mask[::-1, :].copy()
        return image, mask

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        image = np.array(image)
        
        if self.mode == 'train':
            mask_name = img_name.replace('_sat.jpg', '_mask.png')
            mask_path = os.path.join(self.img_dir, mask_name)
            mask = Image.open(mask_path)
            mask = np.array(mask)

            if self.augment:
                image, mask = self.random_flip(image, mask)

            mask = (mask >= 128).astype(int)
            mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]

            masks = np.zeros_like(mask)
            masks[mask == 3] = 0
            masks[mask == 6] = 1
            masks[mask == 5] = 2
            masks[mask == 2] = 3
            masks[mask == 1] = 4
            masks[mask == 7] = 5
            masks[mask == 0] = 6

            if self.transform:
                image = self.transform(Image.fromarray(image))

            return image, torch.from_numpy(masks).long()

        elif self.mode == 'valid':  # 在 valid mode 下返回圖像和名稱
            if self.transform:
                image = self.transform(Image.fromarray(image))
            return image, img_name.replace('_sat.jpg', '_mask.png')

        else:
            if self.transform:
                image = self.transform(Image.fromarray(image))
            return image

class test_plot_Dataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_name = ['0013_sat.jpg','0062_sat.jpg','0104_sat.jpg']   


    def __getitem__(self, index):
        image = self.img_name[index]
        img_path = os.path.join(self.img_dir, image)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        
        return image

    def __len__(self):
        return len(self.img_name)

# 測試 dataloader 是否正常工作
if __name__ == "__main__":
    data_dir = '/home/remote/tchuang/dlcv-fall-2024-hw1-huangchink/hw1_data/p2_data/train/'  # 替換為你的數據路徑
        # 定義圖像的 transforms
    TRANSFORM_IMG = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 在 'train' 模式下測試
    dataset = SegmentationDataset(data_dir, mode='train', transform=TRANSFORM_IMG, augment=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    print(f"Dataset size: {len(dataset)}")
    for i, (images, masks) in enumerate(dataloader):
        print(f"Batch {i + 1} - Image shape: {images.shape}, Mask shape: {masks.shape}")
        if i == 1:
            break

    # 在 'val' 模式下測試
    # dataset_val = SegmentationDataset(data_dir, mode='val', transform=TRANSFORM_IMG, augment=False)
    # dataloader_val = DataLoader(dataset_val, batch_size=4, shuffle=False, num_workers=0)
    # dataset_test = test_plot_Dataset(data_dir, transform=TRANSFORM_IMG)
    # dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)

    # print(f"Validation dataset size: {len(dataset_val)}")
    # for i, (images) in enumerate(dataloader_val):
    #     print(f"Batch {i + 1} - Image shape: {images.shape}")
    #     if i == 1:
    #         break
    # print(f"test dataset size: {len(dataloader_test)}")
    # for i, (images) in enumerate(dataloader_test):
    #     print(f"Batch {i + 1} - Image shape: {images.shape}")
    #     if i == 1:
    #         break