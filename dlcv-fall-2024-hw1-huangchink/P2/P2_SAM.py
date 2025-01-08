import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import os
import numpy as np
import matplotlib.pyplot as plt
from P2_dataloader import test_plot_Dataset  # 假設SegmentationDataset定義在P2_dataloader.py中
from PIL import Image

# 定義展示掩膜的函數，使用固定的 7 種顏色
# 定義展示掩膜的函數，使用固定的 7 種顏色
def show_anns(anns, idx, output_dir):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3), dtype=np.uint8)

    # 定義 7 種顏色
    cls_color = [
        [0, 255, 255],    # Cyan: Urban land
        [255, 255, 0],    # Yellow: Agriculture land
        [255, 0, 255],    # Purple: Rangeland
        [0, 255, 0],      # Green: Forest land
        [0, 0, 255],      # Blue: Water
        [255, 255, 255],  # White: Barren land
        [0, 0, 0]         # Black: Unknown
    ]

    for ann in sorted_anns:
        m = ann['segmentation']
        random_color = np.random.choice(len(cls_color))  # 隨機選擇一種顏色
        img[m] = cls_color[random_color]  # 選定顏色覆蓋到 mask 區域

    output_image = Image.fromarray(img)
    output_image.save(os.path.join(output_dir, f'sam_test_pred_{idx}_colored.png'))


# 主推理腳本
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 選擇 SAM 模型並加載權重
    model_type = "vit_h"  # 你可以選擇 'vit_h', 'vit_l', 或 'vit_b'
    sam_checkpoint = "sam_vit_h_4b8939.pth"  # SAM 預訓練權重的檔案路徑
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    
    # 加載測試圖像
    image_path = ['../hw1_data/p2_data/validation/0013_sat.jpg','../hw1_data/p2_data/validation/0062_sat.jpg','../hw1_data/p2_data/validation/0104_sat.jpg']
    for i in range(len(image_path)):
        image = Image.open(image_path[i])
        image = np.array(image)

        # 定義輸出目錄
        output_dir = 'sam_test_output'
        os.makedirs(output_dir, exist_ok=True)

        # 將 SAM 模型移動到 GPU
        sam.to(device=device)

        # 使用 SAM 自動生成掩膜
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(image)

        # 展示並保存生成的掩膜
        show_anns(masks, idx=i, output_dir=output_dir)

        print(f"Inference complete and results saved in {output_dir}!")


if __name__ == "__main__":
    main()
