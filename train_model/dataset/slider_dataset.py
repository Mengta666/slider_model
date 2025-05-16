import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class SliderDataset(Dataset):
    def __init__(self, image_dir, json_dir, transform=None):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))
        self.json_files = sorted(os.listdir(json_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 加载图像
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")

        # 加载对应的标注
        json_info_path = os.path.join(self.json_dir, self.json_files[idx])
        with open(json_info_path, 'r') as f:
            json_info = json.load(f)

        # 提取边界框和标签
        boxes = []
        labels = []
        shape = json_info['shapes'][0]                     # 列表中只有一个元素，用不用循环都可以
        if shape['shape_type'] == 'rectangle':
            x1, y1 = shape['points'][0]
            x2, y2 = shape['points'][1]
            # 转换为 [x_min, y_min, x_max, y_max]
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(1)  # 假设 'gap' 是类别 1

        # 转换为 tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # 构造目标字典
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),               # 这个标签对应的图像的索引
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),     # 区域面积（大致）
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)        # 边框上面是否有拥挤对象，默认为 0（也就是没有）
        }

        # 应用变换
        if self.transform:
            img = self.transform(img)

        return img, target



if __name__ == '__main__':
    # 数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将 PIL 图像转换为 tensor
    ])
    # 创建数据集
    train_dataset = SliderDataset(
        image_dir = 'image',
        json_dir = "info",
        transform=transform
    )

    print(train_dataset.__getitem__(1000)[1]["image_id"][0])