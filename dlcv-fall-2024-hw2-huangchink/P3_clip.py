import torch
import clip
from PIL import Image
import os
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加載 CLIP 模型
model, preprocess = clip.load("ViT-B/32", device=device)

# 讀取 id2label.json 檔案
with open('hw2_data/clip_zeroshot/id2label.json', 'r') as f:
    id2label = json.load(f)

# 生成文本標籤
labels = [f"A photo of {label}" for label in id2label.values()]
text_inputs = clip.tokenize(labels).to(device)

# 圖像文件夾路徑
image_folder = "hw2_data/clip_zeroshot/val"

# 記錄正確與錯誤分類的數量
correct = 0
total = 0

# 儲存正確與錯誤案例
success_cases = []
failure_cases = []
print('doing zero shot classificatin: ')
# 進行推論
for image_file in os.listdir(image_folder):
    if image_file.endswith(".png"):
        class_id = int(image_file.split('_')[0])  # 獲取正確的 class_id
        image_path = os.path.join(image_folder, image_file)
        
        # 預處理圖像
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        # CLIP 模型推論
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text_inputs)

            logits_per_image, logits_per_text = model(image, text_inputs)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # 預測結果
        predicted_class_id = probs.argmax()
        
        # 檢查是否分類正確
        total += 1
        if predicted_class_id == class_id:
            correct += 1
            success_cases.append((image_file, id2label[str(class_id)]))
        else:
            failure_cases.append((image_file, 'GT:'+id2label[str(class_id)], 'Predicted:'+id2label[str(predicted_class_id)]))

# 計算準確度
accuracy = correct / total * 100
print(f"Accuracy: {accuracy}%")

# 顯示成功和失敗案例
for i in range(5):
    print("Successful Cases:", success_cases[i])
for i in range(5):
    print("Failed Cases:", failure_cases[i])