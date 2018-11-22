# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
import copy

from utils.coco_image_floder import COCOImageFloder

# --------------------路径参数--------------------

test_data_path = "/home/datasets/qishuo/coco/val2017"
# val_data_path = "/Users/qs/Downloads/val2017"
test_label_file_path = "/home/datasets/qishuo/coco/COCO_label_val.txt"
test_batch_size = 10
num_epochs = 100
GPU_id = "cuda:3"
# 待加载模型路径
model_path = "./models/Multi-Class_ResNet_Epoch60_acc0.3121970920840065.pkl"
# 分类时置信度（阈值）
threshold = 0.50

# --------------------加载COCO数据--------------------

print("Getting COCO data...")


transform_val = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


datasets_val = COCOImageFloder(root=test_data_path,
                               label=test_label_file_path,
                               transform=transform_val)

dataLoader_val = torch.utils.data.DataLoader(datasets_val,
                                             batch_size=test_batch_size,
                                             shuffle=True,
                                             num_workers=2)

# --------------------全局参数--------------------

class_name = datasets_val.classes
device = torch.device(GPU_id if torch.cuda.is_available() else "cpu")
val_size = len(datasets_val)

# --------------------模型--------------------

print("Getting model...")

model = torchvision.models.resnet101(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 80)
model.load_state_dict(torch.load(model_path))
model = model.to(device)
# 测试模式，关闭Bn及Dropout
model.eval()

# --------------------测试--------------------

# 统计验证集中正确个数
running_corrects = 0
# 统计存在某一类的图像的个数
all_label_num = [0 for _ in range(80)]
# 统计存在某一类的图像中预测正确的个数
true_label_num = [0 for _ in range(80)]

j = 0
print("Testing...")
for inputs_val, labels_val in dataLoader_val:
    j += 1
    print("iter : " + str(j))
    inputs_val = inputs_val.to(device)
    labels_val = labels_val.to(device)

    logits_val = model(inputs_val)
    outputs_val = torch.nn.Sigmoid()(logits_val)

    preds_val_numpy = outputs_val.cpu().detach().numpy()
    labels_val_numpy = labels_val.cpu().detach().numpy()

    for k in range(len(preds_val_numpy)):
        for i in range(len(preds_val_numpy[k])):
            if labels_val_numpy[k][i] == 1:
                all_label_num[i] += 1
            if labels_val_numpy[k][i] == 1 and preds_val_numpy[k][i] >= threshold:
                true_label_num[i] += 1

print(all_label_num)
print(true_label_num)
for mm in range(80):
    print(class_name[mm] + "    Recall : " + str(float(true_label_num[mm]) / float(all_label_num[mm])))
