#!/usr/bin/env python
# coding: utf-8

# In[17]:


get_ipython().system('jupyter nbconvert --to script pyramidnet_tiny.ipynb')


# In[18]:


from tinygrad.tensor import Tensor
import tinygrad.nn as nn
from tinygrad.nn.datasets import cifar
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.dtype import dtypes
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
from typing import List, Tuple, Optional, Callable
import time
from tinygrad import Device
from tinygrad.helpers import Context


# In[19]:


class BasicBlock():
    outchannel_ratio = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        # eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        # https://docs.tinygrad.org/nn/#tinygrad.nn.BatchNorm
        # eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        self.bn1 = nn.BatchNorm(in_channels)
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # dilation=1, groups=1
        # https://docs.tinygrad.org/nn/#tinygrad.nn.Conv2d
        # dilation=1, groups=1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm(out_channels)
        self.downsample = downsample
        self.stride = stride

    def __call__(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = out.relu()
        out = self.conv2(out)
        
        out = self.bn3(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = Tensor.zeros(batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1])
            out = out + Tensor.cat(shortcut, padding, dim=1)
        else:
            out = out + shortcut
        return out


class Bottleneck():
    outchannel_ratio = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        self.bn1 = nn.BatchNorm(in_planes)
        # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        # stride=1, padding=0
        # https://docs.tinygrad.org/nn/#tinygrad.nn.BatchNorm
        # stride=1, padding=0
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.outchannel_ratio, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm(planes * Bottleneck.outchannel_ratio)
        self.relu = Tensor.relu
        self.downsample = downsample
        self.stride = stride

    def __call__(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        
        out = self.bn4(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = Tensor.zeros(batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1])
            out = out + Tensor.cat(shortcut, padding, dim=1)
        else:
            out = out + shortcut

        return out


class PyramidNet:
    def __init__(self, num_classes, depth, alpha, bottleneck=False):
        if depth not in [18, 34, 50, 101, 152, 200]:
            if bottleneck:
                block = Bottleneck
                temp_cfg = (depth - 2) // 12
            else:
                block = BasicBlock
                temp_cfg = (depth - 2) // 8
            layers = [temp_cfg, temp_cfg, temp_cfg, temp_cfg]
            print('=> the layer configuration for each stage is set to', layers[depth])
        else:
            block = BasicBlock if depth <= 34 and not bottleneck else Bottleneck
            if depth == 18:
                layers = [2, 2, 2, 2]
            elif depth in [34, 50]:
                layers = [3, 4, 6, 3]
            elif depth == 101:
                layers = [3, 4, 23, 3]
            elif depth == 152:
                layers = [3, 8, 36, 3]
            else:
                layers = [3, 24, 36, 3]

        self.in_planes = 64            
        self.addrate = alpha / sum(layers)

        self.input_featuremap_dim = self.in_planes
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        self.featuremap_dim = self.input_featuremap_dim 
        self.layer1 = self.pyramidal_make_layer(block, layers[0])
        self.layer2 = self.pyramidal_make_layer(block, layers[1], stride=2)
        self.layer3 = self.pyramidal_make_layer(block, layers[2], stride=2)
        self.layer4 = self.pyramidal_make_layer(block, layers[3], stride=2)

        self.final_featuremap_dim = self.input_featuremap_dim
        self.bn_final = nn.BatchNorm2d(self.final_featuremap_dim)
        self.avgpool = lambda x: x.avg_pool2d(7)
        self.fc = nn.Linear(self.final_featuremap_dim, num_classes)

    def pyramidal_make_layer(self, block, block_depth, stride=1):
        downsample = None
        if stride != 1:
            downsample = lambda x: x.avg_pool2d((2, 2), stride=(2, 2), ceil_mode=True)

        layers = []
        self.featuremap_dim += self.addrate
        layers.append(block(self.input_featuremap_dim, int(round(self.featuremap_dim)), stride, downsample))
        for i in range(1, block_depth):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            layers.append(block(int(round(self.featuremap_dim)) * block.outchannel_ratio, int(round(temp_featuremap_dim)), 1))
            self.featuremap_dim  = temp_featuremap_dim
        self.input_featuremap_dim = int(round(self.featuremap_dim)) * block.outchannel_ratio
        return layers

    def __call__(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = x.relu()
        x = x.max_pool2d(kernel_size=3, stride=2, padding=1)

        for layer in self.layer1:
            x = layer(x)
        for layer in self.layer2:
            x = layer(x)
        for layer in self.layer3:
            x = layer(x)
        for layer in self.layer4:
            x = layer(x)

        x = self.bn_final(x)
        x = x.relu()
        x = x.avg_pool2d(7)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    


# In[20]:


class TinygradImageFolder:
    def __init__(self, root: str, annotations_file: Optional[str] = None, class_to_idx: Optional[dict] = None, transform: Optional[list] = None):
        self.root = root
        self.annotations_file = annotations_file
        self.transform = transform

        if annotations_file:
            self.class_to_idx = class_to_idx
            self.samples = self._load_validation_annotations()
            self.classes = None
        else:
            self.classes, self.class_to_idx = self._find_classes(root)
            self.samples = self._make_dataset()
            
    def get_class_to_idx(self):
        return self.class_to_idx;

    def _find_classes(self, directory: str) -> Tuple[List[str], dict]:
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self) -> List[Tuple[str, int]]:
        instances = []
        for class_name in self.classes:
            class_index = self.class_to_idx[class_name]
            class_dir = os.path.join(self.root, class_name)
            for root, _, fnames in sorted(os.walk(class_dir)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if self._is_valid_image(path):
                        instances.append((path, class_index))
        return instances

    def _load_validation_annotations(self) -> List[Tuple[str, int]]:
        instances = []
        with open(self.annotations_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            
            image_name, class_name = parts[:2]
            image_path = os.path.join(self.root, image_name)

            class_index = self.class_to_idx[class_name]

            if self._is_valid_image(image_path):
                instances.append((image_path, class_index))

        return instances

    @staticmethod
    def _is_valid_image(path: str) -> bool:
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except (IOError, SyntaxError):
            return False

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        path, class_index = self.samples[index]
        with Image.open(path) as img:
            if self.transform:
                for t in self.transform:
                    img = t(img)
        return Tensor(img), Tensor(class_index)


class TinygradDataLoader:
    def __init__(self, dataset: TinygradImageFolder, batch_size: int, shuffle: bool):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        if shuffle:
            self.shuffle_array = np.arange(len(dataset))
            np.random.shuffle(self.shuffle_array)
        self.current = 0

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current >= len(self.dataset):
            raise StopIteration
        
        begin = self.current
        end = self.current + self.batch_size
        if end > len(self.dataset):
            end = len(self.dataset)
        if self.shuffle == True:
            x = [self.dataset[i][0]
                 for i in self.shuffle_array[begin:end]]
            y = [self.dataset[i][1]
                 for i in self.shuffle_array[begin:end]]
        else:
            x = [self.dataset[i][0]
                for i in range(begin, end)]
            y = [self.dataset[i][1]
                for i in range(begin, end)]
        self.current += self.batch_size
        return Tensor.stack(*x), Tensor.stack(*y)
        


# In[ ]:


print(Device.DEFAULT)

num_classes = 200
depth = 18
alpha = 48
batch_size = 64
epochs = 1000
learning_rate = 0.001

transforms = [
    lambda x: x.resize((224, 224)),
    lambda x: x.convert("RGB"),
    lambda x: np.array(x).transpose((2, 0, 1))
]

train_dataset = TinygradImageFolder(
    root='C:\\Users\\elect\\Documents\\SEM9\\Advanced Machine Learning\\tiny-imagenet-200\\train', transform=transforms)

val_dataset = TinygradImageFolder(
    root='C:\\Users\\elect\\Documents\\SEM9\\Advanced Machine Learning\\tiny-imagenet-200\\val\\images', annotations_file="C:/Users/elect/Documents/SEM9/Advanced Machine Learning/tiny-imagenet-200/val/val_annotations.txt", class_to_idx=train_dataset.get_class_to_idx(), transform=transforms)


train_loader = TinygradDataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
val_loader = TinygradDataLoader(
    val_dataset, batch_size=batch_size, shuffle=False)

model = PyramidNet(num_classes=num_classes, depth=depth, alpha=alpha, bottleneck=True)

optimizer = Adam(get_parameters(model), lr=learning_rate)

with Tensor.train():
    for i, (images, labels) in tqdm(enumerate(train_loader)):
        time1 = time.time()
        
        out = model(images)
        time2 = time.time()
        
        loss = out.sparse_categorical_crossentropy(labels)
        time3 = time.time()
        
        optimizer.zero_grad()
        time4 = time.time()
        
        loss.backward()
        time5 = time.time()
        
        optimizer.step()
        time6 = time.time()

        if i == 500:
            break
        
        if i % 100 == 99:
            pred = out.argmax(axis=-1)
            acc = (pred == labels).mean()
            print(f"Loss: {loss.numpy()} | Accuracy: {acc.numpy()}")

with Tensor.test():
    acc = []
    for val_images, val_labels in tqdm(val_loader):
        out = model(val_images)
        pred = out.argmax(axis=-1)
        acc.append((pred == val_labels).mean().numpy())
    print(sum(acc) / len(acc))


# In[ ]:





# In[ ]:




