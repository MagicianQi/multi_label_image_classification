# -*- coding: utf-8 -*-

"""
由于COCO的图像全部存放在一个文件夹内，没有分类存储，无法使用torchvision提供的ImageFolder加载数据。
所以我们使用label文件来读取标签信息。
"""

import os
import torch
import torch.utils.data as data
from PIL import Image


def default_loader(path):
    return Image.open(path).convert('RGB')


class COCOImageFloder(data.Dataset):
    def __init__(self, root, label, transform=None, target_transform=None, loader=default_loader):
        fh = open(label)
        c = 0
        imgs = []
        class_names = []
        for line in fh.readlines():
            if c == 0:
                class_names = [n.strip() for n in line.strip().split(',')]
            else:
                cls = line.split()
                fn = cls.pop(0)
                if os.path.isfile(os.path.join(root, fn)):
                    imgs.append((fn, tuple([int(v) for v in cls])))
            c = c + 1
        self.root = root
        self.imgs = imgs
        self.classes = class_names
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, fn))
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor(label)

    def __len__(self):
        return len(self.imgs)

    def getName(self):
        return self.classes