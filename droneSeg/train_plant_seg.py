import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

from PIL import Image
import cv2
import albumentations as A

import time
import os
from tqdm.notebook import tqdm

from torchsummary import summary
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import json
from labelme import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PlatSegmentationData(Dataset):
    def __init__(self,image_dir,mask_dir,transform = None):
        self.transforms = transform
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith(".tif")]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        img = np.array(Image.open(os.path.join(self.image_dir, self.images[index])))
    
        mask = np.array(Image.open(os.path.join(
            self.mask_dir, self.images[index].replace(".tif",".png"))))

        mask[mask==2] = 0
        

        # print(mask.shape)
        if self.transforms is not None:
            aug = self.transforms(image=img,mask=mask)
            img = aug['image']
            mask = aug['mask']
            # mask = torch.max(mask,dim=2)[0]

        return img,mask

def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


def mIoU(pred_mask, mask, smooth=1e-10, n_classes=23):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)



t = A.Compose([
    A.Resize(1024,1024),
    A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])



IMAGE_PATH = "/home/haichuan/datasets/OAM-TCD/dataset/holdout/images"
MASK_PATH = "/home/haichuan/datasets/OAM-TCD/dataset/holdout/masks"


train_dataset = PlatSegmentationData(image_dir=os.path.join(IMAGE_PATH,'train'), mask_dir=os.path.join(MASK_PATH,'train'),transform=t)
val_dataset = PlatSegmentationData(image_dir=os.path.join(IMAGE_PATH,'val'), mask_dir=os.path.join(MASK_PATH,'val'),transform=t)

batch_size = 2
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

n_classes = 2

model = smp.DeepLabV3('resnet101', encoder_weights='imagenet', classes=n_classes, activation=None)



LEARNING_RATE = 1e-4
num_epochs = 20

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler()

model = model.to(device)

for epoch in range(num_epochs):
    loop = tqdm(enumerate(train_dl),total=len(train_dl))
    for batch_idx, (data, targets) in loop:
        data = data.to(device)
        targets = targets.to(device)
        targets = targets.type(torch.long)
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


torch.save(model, "plant_seg_drone.pt")