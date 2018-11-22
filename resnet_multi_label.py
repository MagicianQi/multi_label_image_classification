# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
import copy

from utils.coco_image_floder import COCOImageFloder

# --------------------路径参数--------------------

train_data_path = "/home/datasets/qishuo/coco/train2017"
val_data_path = "/home/datasets/qishuo/coco/val2017"
# train_data_path = "/Users/qs/Downloads/train2017"
# val_data_path = "/Users/qs/Downloads/val2017"
train_label_file_path = "/home/datasets/qishuo/coco/COCO_label_train.txt"
val_label_file_path = "/home/datasets/qishuo/coco/COCO_label_val.txt"
train_batch_size = 400
val_batch_size = 400
num_epochs = 300
GPU_id = "cuda:3"
lr_init = 0.01

# --------------------加载COCO数据--------------------

print("Getting COCO data...")

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_val = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

datasets_train = COCOImageFloder(root=train_data_path,
                                 label=train_label_file_path,
                                 transform=transform_train)

datasets_val = COCOImageFloder(root=val_data_path,
                               label=val_label_file_path,
                               transform=transform_val)

dataLoader_train = torch.utils.data.DataLoader(datasets_train,
                                               batch_size=train_batch_size,
                                               shuffle=True,
                                               num_workers=2)

dataLoader_val = torch.utils.data.DataLoader(datasets_val,
                                             batch_size=val_batch_size,
                                             shuffle=True,
                                             num_workers=2)

# --------------------全局参数--------------------

class_name = datasets_train.classes
device = torch.device(GPU_id if torch.cuda.is_available() else "cpu")
train_size = len(datasets_train)
val_size = len(datasets_val)

# --------------------模型--------------------

print("Getting models...")

model = torchvision.models.resnet101(pretrained=True)
# 冻结所有层
for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Linear(model.fc.in_features, 80)
model = model.to(device)
model.train(mode=True)

# --------------------损失函数及优化算法--------------------

criterion = torch.nn.MultiLabelSoftMarginLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_init, betas=(0.9, 0.99))
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

# --------------------训练--------------------

print("Training...")

# 临时保存最佳参数
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 20)
    # 统计训练集loss值
    running_loss = 0.0
    # 统计训练集预测正确个数
    running_corrects = 0
    # 学习率衰减
    exp_lr_scheduler.step()
    # 用于打印iter轮数
    i = 0
    for inputs, labels in dataLoader_train:
        i += 1
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 将上一次迭代的梯度值置零
        optimizer.zero_grad()
        logits = model(inputs)
        # 经过sigmoid得到每一类的置信度
        outputs = torch.nn.Sigmoid()(logits)
        # 计算损失值
        loss = criterion(logits, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 计算阈值为0.5时的预测结果向量
        preds = torch.round(outputs)
        # 预测结果向量与标签向量完全一致时，才认为预测准确
        for tensor_p, tensor_l in zip(preds, labels.data):
            if torch.equal(tensor_p, tensor_l):
                running_corrects += 1
        print("Epoch : " + str(epoch) + "    Iter : " + str(i) + "    Loss : " + str(loss.item())
              + "    Right_Num : " + str(running_corrects))

    train_acc = float(running_corrects) / float(train_size)
    print("train acc  : " + str(train_acc))

    # --------------------验证集与训练集代码基本相同-------------------

    # 统计验证集中正确个数
    running_corrects_val = 0
    j = 0
    print("Val start...")
    for inputs_val, labels_val in dataLoader_val:
        j += 1
        inputs_val = inputs_val.to(device)
        labels_val = labels_val.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            logits_val = model(inputs_val)
            outputs_val = torch.nn.Sigmoid()(logits_val)
            loss_val = criterion(logits_val, labels_val)

        preds_val = torch.round(outputs_val)
        for tensor_p, tensor_l in zip(preds_val, labels_val.data):
            if torch.equal(tensor_p, tensor_l):
                running_corrects_val += 1
        print("Val  Iter : " + str(j) + "    Loss : " + str(loss_val.item())
              + "    Right_Num : " + str(running_corrects_val))

    val_acc = float(running_corrects_val) / float(val_size)
    print("val acc  : " + str(val_acc))

    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())

# 加载最佳模型参数
model.load_state_dict(best_model_wts)
print(best_acc)
# 保存模型
torch.save(model.state_dict(), "./models/Multi-Class_ResNet101_Epoch:" + str(num_epochs) + '_BestAcc:' + str(best_acc) + '.pkl')
