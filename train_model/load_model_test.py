import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np

# 推理时的变换
transform = transforms.Compose([
    transforms.ToTensor(),  # 将 PIL 图像转换为 tensor
])

def load_model(model_path):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(Weights='DEFAULT')
    num_classes = 2  # 1 个前景类（gap） + 背景
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    device = torch.device('cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def draw_box(image, box, output_path):
    # 创建一个 ImageDraw 对象来绘制红框
    draw = ImageDraw.Draw(image)
    # box 格式为 [x_min, y_min, x_max, y_max]
    draw.rectangle(box, outline='red', width=3)
    # 保存带有红框的图像
    image.save(output_path)
    print(f"已保存带 红框图像至: {output_path}")

if __name__ == '__main__':
    model, device = load_model('slider_fasterrcnn.pth')
    image = Image.open(r'E:\python\spider\selenium\picture\douban\cap_union_new_getcapbysig.png').convert('RGB')
    image_tensor = transform(image).to(device)
    with torch.no_grad():
        predictions = model([image_tensor])
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    max_score_idx = np.argmax(scores)
    max_score_box = boxes[max_score_idx]
    max_score = scores[max_score_idx]
    print("最大分数的边界框:", max_score_box)
    print("最大分数:", max_score)

    # 在图像上绘制红框并保存
    draw_box(image, max_score_box, 'output_with_box.png')