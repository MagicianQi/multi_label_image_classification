# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# ---------------------------目标分类---------------------------

# 1.CIFAR-10 10个类别，每类6000张

CIFAR10_classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 2.CIFAR-100 20个超类 每个超类有5个子类，一共100个子类，每个子类600张

CIFAR100_classes = {'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                    'fish': ['aquarium fish', 'flatfish', 'ray', 'shark', 'trout'],
                    'flowers': ['orchids', 'poppies', 'roses', 'sunflowers', 'tulips'],
                    'food containers': ['bottles', 'bowls', 'cans', 'cups', 'plates'],
                    'fruit and vegetables': ['apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers'],
                    'household electrical devices': ['clock', 'computer keyboard', 'lamp', 'telephone', 'television'],
                    'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                    'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                    'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                    'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                    'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                    'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                    'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
                    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
                    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                    'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                    'trees': ['maple', 'oak', 'palm', 'pine', 'willow'],
                    'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup truck', 'train'],
                    'vehicles 2': ['lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor']}

# 3.MS-COCO，图像内有多个目标 90个类别

MS_COCO_classes = {'person': [],
                   'vehicle': ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
                   'outdoor': ["traffic light", "fire hydrant", "stop sign", "parking meter", "bench"],
                   'animal': ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
                   'accessory': ["backpack", "umbrella", "handbag", "tie", "suitcase"],
                   'sports': ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                              "skateboard", "surfboard", "tennis racket"],
                   'kitchen': ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"],
                   'food': ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
                            "cake"],
                   'furniture': ["chair", "couch", "potted plant", "bed", "dining table", "toilet"],
                   'electronic': ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone"],
                   'appliance': ["microwave", "oven", "toaster", "sink", "refrigerator"],
                   'indoor': ["book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]}

# 4.ImageNet-12, 1000个类别，每类大约150张

ImageNet_12_classes = []
with open('ImageNet-12 classes.txt', 'r') as file:
    for line in file.readlines():
        ImageNet_12_classes.append(line.strip())
# print(ImageNet_12_classes)


# 5.Open Images v4(Google), 1.9W+类别，每类大约1000张

Open_Images_v4_classes = []

with open('Open Images class-descriptions.csv', 'r') as file:
    for line in file.readlines():
        Open_Images_v4_classes.append(line.strip().split(",")[1])
# print(Open_Images_v4_classes)


# 6.PASCAL VOC 07, 20个类别 每类大约500张

PASCAL_VOC_07_classes = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'airplane', 'bicycle', 'boat',
                         'bus', 'car', 'motorbike', 'train', 'bottle', 'chair', 'dining table', 'potted plant',
                         'sofa', 'monitor']

# 7.Caltech 256, 256个类别，每类大约150张

Caltech256_classes = ['ak47', 'american-flag', 'backpack', 'baseball-bat', 'baseball-glove', 'basketball-hoop', 'bat',
                      'bathtub', 'bear', 'beer-mug', 'billiards', 'binoculars', 'birdbath', 'blimp', 'bonsai-101',
                      'boom-box', 'bowling-ball', 'bowling-pin', 'boxing-glove', 'brain-101', 'breadmaker',
                      'buddha-101', 'bulldozer', 'butterfly', 'cactus', 'cake', 'calculator', 'camel', 'cannon',
                      'canoe', 'car-tire', 'cartman', 'cd', 'centipede', 'cereal-box', 'chandelier-101', 'chess-board',
                      'chimp', 'chopsticks', 'cockroach', 'coffee-mug', 'coffin', 'coin', 'comet', 'computer-keyboard',
                      'computer-monitor', 'computer-mouse', 'conch', 'cormorant', 'covered-wagon', 'cowboy-hat',
                      'crab-101', 'desk-globe', 'diamond-ring', 'dice', 'dog', 'dolphin-101', 'doorknob',
                      'drinking-straw', 'duck', 'dumb-bell', 'eiffel-tower', 'electric-guitar-101', 'elephant-101',
                      'elk', 'ewer-101', 'eyeglasses', 'fern', 'fighter-jet', 'fire-extinguisher', 'fire-hydrant',
                      'fire-truck', 'fireworks', 'flashlight', 'floppy-disk', 'football-helmet', 'french-horn',
                      'fried-egg', 'frisbee', 'frog', 'frying-pan', 'galaxy', 'gas-pump', 'giraffe', 'goat',
                      'golden-gate-bridge', 'goldfish', 'golf-ball', 'goose', 'gorilla', 'grand-piano-101', 'grapes',
                      'grasshopper', 'guitar-pick', 'hamburger', 'hammock', 'harmonica', 'harp', 'harpsichord',
                      'hawksbill-101', 'head-phones', 'helicopter-101', 'hibiscus', 'homer-simpson', 'horse',
                      'horseshoe-crab', 'hot-air-balloon', 'hot-dog', 'hot-tub', 'hourglass', 'house-fly',
                      'human-skeleton', 'hummingbird', 'ibis-101', 'ice-cream-cone', 'iguana', 'ipod', 'iris',
                      'jesus-christ', 'joy-stick', 'kangaroo-101', 'kayak', 'ketch-101', 'killer-whale', 'knife',
                      'ladder', 'laptop-101', 'lathe', 'leopards-101', 'license-plate', 'lightbulb', 'light-house',
                      'lightning', 'llama-101', 'mailbox', 'mandolin', 'mars', 'mattress', 'megaphone', 'menorah-101',
                      'microscope', 'microwave', 'minaret', 'minotaur', 'motorbikes-101', 'mountain-bike', 'mushroom',
                      'mussels', 'necktie', 'octopus', 'ostrich', 'owl', 'palm-pilot', 'palm-tree', 'paperclip',
                      'paper-shredder', 'pci-card', 'penguin', 'people', 'pez-dispenser', 'photocopier', 'picnic-table',
                      'playing-card', 'porcupine', 'pram', 'praying-mantis', 'pyramid', 'raccoon', 'radio-telescope',
                      'rainbow', 'refrigerator', 'revolver-101', 'rifle', 'rotary-phone', 'roulette-wheel', 'saddle',
                      'saturn', 'school-bus', 'scorpion-101', 'screwdriver', 'segway', 'self-propelled-lawn-mower',
                      'sextant', 'sheet-music', 'skateboard', 'skunk', 'skyscraper', 'smokestack', 'snail', 'snake',
                      'sneaker', 'snowmobile', 'soccer-ball', 'socks', 'soda-can', 'spaghetti', 'speed-boat', 'spider',
                      'spoon', 'stained-glass', 'starfish-101', 'steering-wheel', 'stirrups', 'sunflower-101',
                      'superman', 'sushi', 'swan', 'swiss-army-knife', 'sword', 'syringe', 'tambourine', 'teapot',
                      'teddy-bear', 'teepee', 'telephone-box', 'tennis-ball', 'tennis-court', 'tennis-racket',
                      'theodolite', 'toaster', 'tomato', 'tombstone', 'top-hat', 'touring-bike', 'tower-pisa',
                      'traffic-light', 'treadmill', 'triceratops', 'tricycle', 'trilobite-101', 'tripod', 't-shirt',
                      'tuning-fork', 'tweezer', 'umbrella-101', 'unicorn', 'vcr', 'video-projector', 'washing-machine',
                      'watch-101', 'waterfall', 'watermelon', 'welding-mask', 'wheelbarrow', 'windmill', 'wine-bottle',
                      'xylophone', 'yarmulke', 'yo-yo', 'zebra', 'airplanes-101', 'car-side-101', 'faces-easy-101',
                      'greyhound', 'tennis-shoes', 'toad', 'clutter']

# 8.Animals with attributes2, 50个类别，每类大约1500张

Animals_with_attributes2_classes = ['antelope', 'grizzly+bear', 'killer+whale', 'beaver', 'dalmatian', 'persian+cat',
                                    'horse', 'german+shepherd', 'blue+whale', 'siamese+cat', 'skunk', 'mole', 'tiger',
                                    'hippopotamus', 'leopard', 'moose', 'spider+monkey', 'humpback+whale', 'elephant',
                                    'gorilla', 'ox', 'fox', 'sheep', 'seal', 'chimpanzee', 'hamster', 'squirrel',
                                    'rhinoceros', 'rabbit', 'bat', 'giraffe', 'wolf', 'chihuahua', 'rat', 'weasel',
                                    'otter', 'buffalo', 'zebra', 'giant+panda', 'deer', 'bobcat', 'pig', 'lion',
                                    'mouse', 'polar+bear', 'collie', 'walrus', 'raccoon', 'cow', 'dolphin']

# 9.Stanford Dogs Dataset, 120个类别 每类大约150张

Stanford_Dogs_classes = ['Affenpinscher', 'Afghan hound', 'African hunting dog', 'Airedale',
                         'American Staffordshire terrier', 'Appenzeller', 'Australian terrier', 'Basenji', 'Basset',
                         'Beagle', 'Bedlington terrier', 'Bernese mountain dog', 'Black-and-tan coonhound',
                         'Blenheim spaniel', 'Bloodhound', 'Bluetick', 'Border collie', 'Border terrier', 'Borzoi',
                         'Boston bull', 'Bouvier des Flandres', 'Boxer', 'Brabancon griffon', 'Briard',
                         'Brittany spaniel', 'Bull mastiff', 'Cairn', 'Cardigan', 'Chesapeake Bay retriever',
                         'Chihuahua', 'Chow', 'Clumber', 'Cocker spaniel', 'Collie', 'Curly-coated retriever',
                         'Dandie Dinmont', 'Dhole', 'Dingo', 'Doberman', 'English foxhound', 'English setter',
                         'English springer', 'EntleBucher', 'Eskimo dog', 'Flat-coated retriever', 'French bulldog',
                         'German shepherd', 'German short-haired pointer', 'Giant schnauzer', 'Golden retriever',
                         'Gordon setter', 'Great Dane', 'Great Pyrenees', 'Greater Swiss Mountain dog', 'Groenendael',
                         'Ibizan hound', 'Irish setter', 'Irish terrier', 'Irish water spaniel', 'Irish wolfhound',
                         'Italian greyhound', 'Japanese spaniel', 'Keeshond', 'Kelpie', 'Kerry blue terrier',
                         'Komondor', 'Kuvasz', 'Labrador retriever', 'Lakeland terrier', 'Leonberg', 'Lhasa',
                         'Malamute', 'Malinois', 'Maltese dog', 'Mexican hairless', 'Miniature pinscher',
                         'Miniature poodle', 'Miniature schnauzer', 'Newfoundland', 'Norfolk terrier',
                         'Norwegian elkhound', 'Norwich terrier', 'Old English sheepdog', 'Otterhound', 'Papillon',
                         'Pekinese', 'Pembroke', 'Pomeranian', 'Pug', 'Redbone', 'Rhodesian ridgeback', 'Rottweiler',
                         'Saint Bernard', 'Saluki', 'Samoyed', 'Schipperke', 'Scotch terrier', 'Scottish deerhound',
                         'Sealyham terrier', 'Shetland sheepdog', 'Shih-Tzu', 'Siberian husky', 'Silky terrier',
                         'Soft-coated wheaten terrier', 'Staffordshire bullterrier', 'Standard poodle',
                         'Standard schnauzer', 'Sussex spaniel', 'Tibetan mastiff', 'Tibetan terrier', 'Toy poodle',
                         'Toy terrier', 'Vizsla', 'Walker hound', 'Weimaraner', 'Welsh springer spaniel',
                         'West Highland white terrier', 'Whippet', 'Wire-haired fox terrier', 'Yorkshire terrier']

# ---------------------------场景分类---------------------------

# 1.Places365, 365个类别

Places365_classes = []
with open('Places365 classes.txt', 'r') as file:
    for line in file.readlines():
        s_data = line.strip().split(" ")[0].strip().split("/")
        if len(s_data) == 3:
            class_name = s_data[-1]
        elif len(s_data) == 4:
            class_name = s_data[-2] + " " + s_data[-1]
        else:
            pass
        Places365_classes.append(class_name)
# print(Places365_classes)


# 2.Sun 397, 397个类别

Sun397_classes = []
data = pd.read_excel("Sun 397 three_levels classes.xlsx", usecols=[0])
for item in np.array(data):
    s_data = item[0].strip().split("/")
    if len(s_data) == 3:
        class_name = s_data[-1]
    elif len(s_data) == 4:
        class_name = s_data[-2] + " " + s_data[-1]
    else:
        pass
    Sun397_classes.append(class_name)
# print(Sun397_classes)


# 3.MIT indoor, 一共67个类别 每个类别100张左右
MIT_indoor_classes = ['kitchen', 'operating_room', 'restaurant_kitchen', 'videostore', 'poolinside', 'mall',
                      'kindergarden', 'buffet', 'hospitalroom', 'library', 'inside_bus', 'bar', 'dentaloffice',
                      'office', 'computerroom', 'grocerystore', 'cloister', 'concert_hall', 'jewelleryshop',
                      'laundromat', 'warehouse', 'gym', 'lobby', 'meeting_room', 'garage', 'inside_subway',
                      'restaurant', 'children_room', 'corridor', 'hairsalon', 'bookstore', 'movietheater', 'elevator',
                      'stairscase', 'artstudio', 'bathroom', 'gameroom', 'locker_room', 'nursery', 'waitingroom',
                      'winecellar', 'florist', 'closet', 'clothingstore', 'pantry', 'prisoncell', 'shoeshop', 'museum',
                      'fastfood_restaurant', 'auditorium', 'subway', 'classroom', 'laboratorywet', 'deli', 'tv_studio',
                      'bedroom', 'bowling', 'livingroom', 'dining_room', 'trainstation', 'airport_inside',
                      'church_inside', 'toystore', 'casino', 'bakery', 'greenhouse', 'studiomusic']
                      'restaurant', 'children_room', 'corridor', 'hairsalon', 'bookstore', 'movietheater', 'elevator',
