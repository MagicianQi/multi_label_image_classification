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
# 统计正确个数中标签全0的个数
all_zero = 0

j = 0
print("Testing...")
for inputs_val, labels_val in dataLoader_val:
    j += 1
    inputs_val = inputs_val.to(device)
    labels_val = labels_val.to(device)

    logits_val = model(inputs_val)
    outputs_val = torch.nn.Sigmoid()(logits_val)

    preds_val_numpy = outputs_val.cpu().detach().numpy()
    labels_val_numpy = labels_val.cpu().detach().numpy()

    for k in range(len(preds_val_numpy)):
        # 标记是否有预测为1，但标签为0的，如果有则预测错误。
        flag = 0
        # 标记是否存在预测标签全是0的
        zero_flag = 0
        for i in range(len(preds_val_numpy[k])):
            if preds_val_numpy[k][i] >= threshold and labels_val_numpy[k][i] == 0:
                flag = 1
            if preds_val_numpy[k][i] >= threshold:
                zero_flag = 1
        if flag == 0:
            running_corrects += 1
        if zero_flag == 0:
            all_zero += 1

    print("iter : " + str(j) + "    num : " + str(running_corrects) + "    zero : " + str(all_zero))


val_acc = float(running_corrects) / float(val_size)
zero_acc = float(all_zero) / float(running_corrects)
print("--------------------------")
print("val acc  : " + str(val_acc))
print("--------------------------")
print("zero acc  : " + str(zero_acc))
print("--------------------------")
