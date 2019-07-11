# 分类可用数据集调研：

计算视觉数据集集合：http://www.cvpapers.com/datasets.html

## 目标分类：

### 1.CIFAR-10

    - 图像总量：60000
    - 类别数量：10 (飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车)
    - 下载方式：torchvision.datasets 接口
    - 特点：图像较单一，通常图像内只有一个目标。
    
### 2.CIFAR-100

    - 图像总量：60000
    - 类别数量：100
    - 下载方式：torchvision.datasets 接口
    - 特点：图像较单一，通常图像内只有一个目标，每类图像较少。
    
### 3.MS-COCO

    - 图像总量：32W+
    - 类别数量：90
    - 下载方式：torchvision.datasets 接口
    - 特点：图像场景更贴近日常，图像内容更复杂，每类图像较多。
    
### 4.ImageNet-12

    - 图像总量：1400W+
    - 类别数量：22000
    - 下载方式：torchvision.datasets 接口
    - 特点：标注不是特别准确但是涵盖类别很多。
    
### 5.Open Images v4(Google)

    - 图像总量：3000W+
    - 类别数量：19794
    - 下载方式：官网 https://storage.googleapis.com/openimages/web/index.html
    - 特点：图像场景复杂，与ImageNet相比标准更加准确。

### 6.PASCAL VOC

    - 图像总量：11530
    - 类别数量：20
    - 下载方式：官网 http://host.robots.ox.ac.uk/pascal/VOC/
    - 特点：标注准确但是类别较少，图像较少。


### 7.Caltech 256

    - 图像总量：30607
    - 类别数量：256
    - 下载方式：官网 http://www.vision.caltech.edu/Image_Datasets/Caltech256/
    - 特点：每类图像较少，图像较老。

### 8.Animals with attributes2

    - 图像总量：37332
    - 类别数量：50
    - 下载方式：官网 https://cvml.ist.ac.at/AwA2/
    - 特点：50类动物的数据集。

### 9.Stanford Dogs Dataset

    - 图像总量：20580
    - 类别数量：120a
    - 下载方式：官网 http://vision.stanford.edu/aditya86/ImageNetDogs/
    - 特点：120类狗分类数据集。
    
### 10.Flower classification datasets

    - 图像总量：1360
    - 类别数量：17
    - 下载方式：官网 http://www.robots.ox.ac.uk/~vgg/data0.html
    - 特点：花分类数据集
    
## 场景分类：

### 1.Places 365

    - 图像总量：130519
    - 类别数量：400+
    - 下载方式：官网 http://places2.csail.mit.edu/download.html
    - 论文：《Places: An Image Database for Deep Scene Understanding》
    - 特点：数据量大，数据类别多。

### 2.SUN 397

    - 图像总量：1000W+
    - 类别数量：899(其中397个较准确)
    - 下载方式：官网 https://vision.princeton.edu/projects/2010/SUN/
    - 论文：《SUN Database: Scene Categorization Benchmark》
    - 特点：数据量一般，数据类别多。

### 3.MIT indoor 67

    - 图像总量：15620
    - 类别数量：67
    - 下载方式：官网 http://web.mit.edu/torralba/www/indoor.html
    - 论文：《Recognizing Indoor Scenes》
    - 特点：数据量较少，仅限于室内场景。

### 4.Scene 15

    - 图像总量：不详
    - 类别数量：15
    - 下载方式：不详
    - 论文：《Beyond Bags of Features: Spatial Pyramid Matching for Recognizing Natural Scene Categories》
    - 特点：不详
