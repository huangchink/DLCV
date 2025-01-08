from P2.P2_dataloader import SegmentationDataset
from P2.P2_model_B import DeepLabV3Model
from PIL import Image
import torch
import numpy as np
import os
from torchvision import transforms
import argparse
from torch.utils.data import DataLoader

# 定義推理函數
def inference(model, dataloader, device, output_folder):
    model.to(device)
    model.eval()

    #os.makedirs(output_folder, exist_ok=True)  # 創建輸出文件夾

    with torch.no_grad():
        for images, img_name in dataloader:
            images = images.to(device)
            outputs = model(images)
            
            # 檢查是否是 OrderedDict
            if isinstance(outputs, dict):
                preds = outputs['out'].max(1)[1].cpu().numpy()  # 取得預測的分類結果
            else:
                preds = outputs.max(1)[1].cpu().numpy()

            # 將每張圖片處理並保存
            for i in range(preds.shape[0]):
                pred = preds[i]
                image = np.zeros((512, 512, 3), dtype=np.uint8)
                image[pred == 0] = [0, 255, 255]      # Cyan: Urban land
                image[pred == 1] = [255, 255, 0]      # Yellow: Agriculture land
                image[pred == 2] = [255, 0, 255]      # Purple: Rangeland
                image[pred == 3] = [0, 255, 0]        # Green: Forest land
                image[pred == 4] = [0, 0, 255]        # Blue: Water
                image[pred == 5] = [255, 255, 255]    # White: Barren land
                image[pred == 6] = [0, 0, 0]          # Black: Unknown

                output_image = Image.fromarray(image)
                output_image.save(os.path.join(output_folder, img_name[i]))  # 使用 img_name 保存結果

    print(f"Inference complete. Results saved to {output_folder}")

if __name__ == "__main__":
    # 使用 argparse 來讓用戶選擇參數
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_image_dir", type=str, required=True, help="Path to the directory containing test images")
    parser.add_argument("--output_image_folder", type=str, required=True, help="Folder to save output images")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加載模型
    model_type = 'resnet101'
    model = DeepLabV3Model(model_type=model_type, num_classes=7).to(device)
    model.load_state_dict(torch.load('DeepLabV3Model_resnet101_7519.pth', map_location=device))  # 加載訓練好的權重

    # 定義數據轉換
    transforms_validation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加載測試集
    val_dataset = SegmentationDataset(img_dir=args.test_image_dir, transform=transforms_validation, mode='valid')  # 使用 valid 模式
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 進行推理
    inference(model, val_loader, device, args.output_image_folder)
