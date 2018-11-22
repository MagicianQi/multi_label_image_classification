# Multi-label Image Classification

* 多目标多分类基础模型，模型采用ResNet_101，ImageNet预训练。
* 训练集与验证集采用MS COCO。

## Pre-requisite

* python
* torch
* torchvision
* numpy
* PIL
* matplotlib(如果需要展示图像)

## Data

使用MS COCO数据集。

图像处理方式：

1. 随机裁剪
2. 随机翻转
3. 归一化

训练集位置：
    
    "ip:/home/datasets/qishuo/coco/train2017"

验证集位置：

    "ip:/home/datasets/qishuo/coco/val2017"

## How to use
修改路径参数、超参数等统一：`vim xxx.python`
1. Clone：
    * `git clone https://github.com/MagicianQi/multi_label_image_classification`
    * `cd ./multi_label_image_classification`
2. 进入虚拟环境(在172.30.1.118上)：
    * `source /home/qishuo/venv/bin/activate`
3. 生成MS COCO分类标签：
    * 生成标签文件：`python ./scrpits/generate_coco_labels.py`  
4. 训练：
    * 训练 ：`python resnet_multi_label.py`
    * 后台训练：`screen python resnet_multi_label.py`
5. 验证：
    * 计算准确率(acc)：`python evaluate_multi_label_acc.py`
    * 计算召回率(recall)：`python evaluate_multi_label_recall.py`
6. 测试：
    * 测试图像结果：`python testing_multi_label.py 1.jpg 2.jpg 3.jpg`

## 展示
Image:

<img src="./test_imgs/5711804442_1d88779ff6_o.jpg" width="200" height="300">

Result(Top5):

['person', 0.9997881], ['cell phone', 0.7692987], ['tv', 0.6630157], ['laptop', 0.5872826], ['tie', 0.24580538]
