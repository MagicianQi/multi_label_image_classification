# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
# import matplotlib.pyplot as plt
import numpy as np
import PIL
import sys

# --------------------路径参数--------------------

GPU_id = "cuda:3"
model_path = "./models/Multi-Class_ResNet_Epoch60_acc0.3121970920840065.pkl"

# --------------------全局参数--------------------

class_name = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
              'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
              'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
              'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
              'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
              'teddy bear', 'hair drier', 'toothbrush']

device = torch.device(GPU_id if torch.cuda.is_available() else "cpu")

# --------------------模型--------------------

print("Getting models...")

model = torchvision.models.resnet101(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 80)
model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
model = model.to(device)
model.eval()

# --------------------显示函数（如果没图形界面，会报错）--------------------


# def imshow(inp, title=None):
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)


# --------------------测试--------------------

print("Testing...")

print("Transform Image...")

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

for filepath in sys.argv[1:]:
    # 加载图像
    image = PIL.Image.open(filepath)
    image = transform(image).float()
    # 升为4维，否则会报错
    image = image.unsqueeze(0)

    image = image.to(device)
    logits = model(image)
    outputs = torch.nn.Sigmoid()(logits)
    preds_numpy = outputs.cpu().detach().numpy()[0]
    result_dict = []
    for name, p in zip(class_name, preds_numpy):
        result_dict.append([name, p])
    result_dict = sorted(result_dict, key=lambda k: k[1], reverse=True)
    print(filepath + " : ")
    print(result_dict)

    # imshow(image.cpu().data[0])
