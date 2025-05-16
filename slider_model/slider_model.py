from flask import Flask, request, jsonify, send_file, render_template
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import os
import io
import json
from datetime import datetime

# 创建 Flask 应用实例，指定模板文件夹为 'page'（位于 slider_model/page/）
slider_model_pre = Flask(__name__, template_folder='page')

# 定义图像转换
transform = transforms.Compose([
    transforms.ToTensor(),
])

def load_model(model_path):
    # 加载 Faster R-CNN 模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    num_classes = 2  # 1 个前景类（gap） + 背景
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    return model, device

def draw_box(image, box):
    # 在图像上绘制红框
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline='red', width=3)
    return image

# 启动时加载模型（模型文件位于 slider_model/slider_fasterrcnn.pth）
model, device = load_model('slider_fasterrcnn.pth')

# 确保上传、输出和保存目录存在（位于 slider_model/ 下）
UPLOAD_FOLDER = 'Uploads'
OUTPUT_FOLDER = 'static/output'
SAVED_IMAGES_FOLDER = 'saved_images'
SAVED_ANNOTATIONS_FOLDER = 'saved_coco_json'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(SAVED_IMAGES_FOLDER, exist_ok=True)
os.makedirs(SAVED_ANNOTATIONS_FOLDER, exist_ok=True)

@slider_model_pre.route('/')
def index():
    # 渲染上传页面，从 slider_model/page/index.html 加载
    return render_template('index.html')

@slider_model_pre.route('/predict', methods=['POST'])
def predict():
    # 检查是否上传了文件
    if 'file' not in request.files:
        return jsonify({'error': '未上传文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    try:
        # 保存上传的图像到 slider_model/uploads/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_path = os.path.join(UPLOAD_FOLDER, f'input_{timestamp}.png')
        image = Image.open(file.stream).convert('RGB')
        image.save(input_path)

        # 转换图像并进行预测
        image_tensor = transform(image).to(device)
        with torch.no_grad():
            predictions = model([image_tensor])

        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()

        if len(scores) == 0:
            return jsonify({'error': '未检测到目标'}), 400

        # 获取最高分数的边界框
        max_score_idx = np.argmax(scores)
        max_score_box = boxes[max_score_idx]
        max_score = float(scores[max_score_idx])

        # 绘制红框并保存输出图像到 slider_model/static/output/
        output_image = draw_box(image, max_score_box)
        output_filename = f'output_{timestamp}.png'
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        output_image.save(output_path)

        # 准备响应数据
        box_coords = {
            'x_min': float(max_score_box[0]),
            'y_min': float(max_score_box[1]),
            'x_max': float(max_score_box[2]),
            'y_max': float(max_score_box[3])
        }

        return jsonify({
            'box': box_coords,
            'score': max_score,
            'image_url': f'/static/output/{output_filename}',
            'input_path': input_path,
            'output_path': output_path
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@slider_model_pre.route('/save_result', methods=['POST'])
def save_result():
    try:
        data = request.get_json()
        input_path = data['input_path']
        output_path = data['output_path']
        box = data['box']
        score = data['score']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存图像到 slider_model/saved_images/
        saved_image_path = os.path.join(SAVED_IMAGES_FOLDER, f'{timestamp}.png')
        saved_image = Image.open(output_path)  # 加载输出图像
        saved_image.save(saved_image_path)
        # 删除output下面多余的图片
        os.remove(output_path)
        # 删除输入图像
        os.remove(input_path)

        # 生成 COCO 格式的 JSON
        coco_annotation = {
            "version": "4.5.7",
            "flags": {},
            "shapes": [
                {
                    "label": "gap",
                    "points": [
                        [box['x_min'], box['y_max']],
                        [box['x_max'], box['y_min']]
                    ],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                }
            ],
            "imagePath": f'{timestamp}.png',
            "imageData": None,
            "imageHeight": saved_image.height,  # 使用保存的图像尺寸
            "imageWidth": saved_image.width
        }

        # 保存 COCO JSON 到 slider_model/saved_coco_json/
        coco_json_path = os.path.join(SAVED_ANNOTATIONS_FOLDER, f'{timestamp}.json')
        with open(coco_json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_annotation, f, ensure_ascii=False, indent=2)

        return jsonify({'message': f'图像和标注已保存至 {saved_image_path} 和 {coco_json_path}'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@slider_model_pre.route('/static/output/<filename>')
def serve_output_image(filename):
    # 提供输出图像（从 slider_model/static/output/）
    return send_file(os.path.join(OUTPUT_FOLDER, filename))

if __name__ == '__main__':
    slider_model_pre.run(host='0.0.0.0', port=5000)