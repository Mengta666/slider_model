# slider_model
一个可以用来预测滑块验证码中滑块位置的模型
## 引言
很多时候当你爬取某些公共数据时，也会促发且不限于一种验证方式，其中，滑块验证与点击验证较为繁琐。前段时间用opencv中特征提取后进行边框模板比对来得到滑块的大致位置，但当滑块背景图复杂度高时，会出现位置偏差较大，或者根本就在其他位置的情况。方才想到用深度学习来训练已知的一些滑块数据集。

本文模型使用fasterrcnn-resnet50-fpn预训练模型，利用对预训练权重进行微调，从原本81类缩减到2类，得到了比较不错的结果。

## 预览
网站直接使用：https://slider_pre.mengta.link/

原图像：

![00000](https://imgur.mengta.link/images/2025/05/16/00000.jpeg)

预测后的：

![00000 预测](https://imgur.mengta.link/images/2025/05/16/00000_.png)

另一种用opencv并不能准确预测的情况（大红框）：

[![match picture](https://imgur.mengta.link/images/2025/05/16/match_picture.png)](https://imgur.mengta.link/image/RuTp)

使用本文模型的预测结果（小红框）：

[![match picture better](https://imgur.mengta.link/images/2025/05/16/match_picture_better.png)](https://imgur.mengta.link/image/Rupx)

## 部署方式（只有docker）：
### 1. 下载本页的所有文件到你的服务器中
### 2. 去我的谷歌云盘下载打包好的模型与前后端调用代码
谷歌云盘下载地址：https://drive.google.com/file/d/1oJKObrPpepf6urYaYJxJOmyRZn7vE8OU/view?usp=sharing
### 3. root权限执行脚本 install_slider_model.sh
### 4. 当然，你也可以根据我写的dockerfile、docker-compose自定义安装

### 缺点：由于数据集有限。只能对于单个缺口进行检测，不过就目前而言（滑块都是水平或者垂直移动，完全可以对获取到的图像进行裁剪后检测，所得到的位置在加上裁剪去的长度就是对应需要移动的距离），后续将尝试多滑块缺口的训练


 假如你需要数据集，请联系我的邮箱：mengta6664@gmail.com
