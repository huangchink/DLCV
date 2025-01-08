import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms

class P1Dataset(Dataset):
    def __init__(self, image_root, transform=None):
        """
        Args:
            image_root (str): Path to the folder containing images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_root = image_root
        self.image_filenames = sorted(os.listdir(image_root))
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_root, self.image_filenames[idx])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.image_filenames[idx]

if __name__ == "__main__":
    image_root = "/home/tchuang/dlcv-fall-2024-hw3-huangchink/hw3_data/p1_data/images/val"
    batch_size = 1

    # Define a default transform to convert PIL image to tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = P1Dataset(image_root=image_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for images, filenames in dataloader:
        print(f"Batch of images loaded: {[name for name in filenames]}")