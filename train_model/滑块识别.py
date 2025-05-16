import dataset.slider_dataset as sd
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
import torchvision
import torch

def collate_fn(batch):
    """
    多个数据样本合并
    """
    return tuple(zip(*batch))

if __name__ == '__main__':
    torch.cuda.empty_cache()
    # 数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将 PIL 图像转换为 tensor
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip()
    ])
    slider_dataset = sd.SliderDataset(
        image_dir='dataset/image',
        json_dir="dataset/info",
        transform=transform
    )

    # 将数据集划分成小批次
    train_loader = DataLoader(
        dataset=slider_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    # 加载模型，使用fasterrcnn-resnet50-fpn预训练模型，默认输出81个类（80种边框图上的类别，1个背景图），本次训练边框图上只有一个类
    # 使用预训练模型提取图像特征，生成 ROI 特征，用于后续的 RPN 和分类/回归任务，无需指定类别数量。
    # 其实这个模型是为81个类设计的，所以需要改变模型结构，若不需要预训练pretrained，将输出的类别数量改为2，num_classes=2
    slider_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)

    # 在前面如不指定类为2（需要预训练），需要进行如下微调，保证最后只有两个类
    num_classes = 2
    in_features = slider_model.roi_heads.box_predictor.cls_score.in_features
    slider_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 使用GPU训练
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    slider_model.to(device)

    # 优化，在图像识别方面，SGD效果一般比Adam效果好
    # 获取所有需要调整的参数（也就是需要梯度下降的参数）
    params = [p for p in slider_model.parameters() if p.requires_grad]
    # 随机梯度下降优化器
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # 学习率调度器，在优化器种调整学习率，默认每3轮调整一次：lr*gamma
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"开始训练：轮次 {epoch + 1}/{num_epochs}")
        slider_model.train()
        total_batches = len(train_loader)  # 总批次数
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = slider_model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # 计算并显示进度
            progress = (batch_idx + 1) / total_batches * 100
            print(f"轮次 {epoch + 1}/{num_epochs}, 批次 {batch_idx + 1}/{total_batches}, 进度: {progress:.2f}%, 损失: {losses.item():.4f}")

        lr_scheduler.step()
        print(f"轮次 {epoch + 1}/{num_epochs} 完成")

    # 保存模型
    torch.save(slider_model.state_dict(), 'slider_fasterrcnn.pth')