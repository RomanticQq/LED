import os, sys, glob, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

import time
import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt

train_df = pd.read_csv('./LED_part1_train/train.csv', sep=' ')
train_df['file'] = train_df['file'].apply(lambda x: str(x).zfill(6) + '.bmp')
train_df = train_df.sample(frac=1.0)

train_path = './LED_part1_train/imgs/' + train_df['file']
train_label = train_df['label']

test_df = pd.read_csv('提交示例.csv')
test_df['path'] = test_df['file'].apply(lambda x: str(x.split('_')[1]).zfill(6) + '.bmp')
test_path = './LED_part1_train/imgs/' + test_df['path']

set(test_path) & set(train_label)


class XunFeiDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = cv2.imread(self.img_path[index])
        img = img.astype(np.float32)

        img /= 255.0
        img -= 1

        if self.transform is not None:
            img = self.transform(image=img)['image']
        img = img.transpose([2, 0, 1])
        return img, torch.from_numpy(np.array(self.img_label[index]))

    def __len__(self):
        return len(self.img_path)


class XunFeiNet(nn.Module):
    def __init__(self):
        super(XunFeiNet, self).__init__()

        model = models.resnet18(True)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(512, 4)
        self.resnet = model

    def forward(self, img):
        out = self.resnet(img)
        return out


import albumentations as A

train_loader = torch.utils.data.DataLoader(
    XunFeiDataset(train_path.values[:-500], train_label.values[:-500],
                  A.Compose([
                      # A.Resize(300, 300),
                      A.RandomCrop(130, 450),
                      A.HorizontalFlip(p=0.5),
                      A.RandomContrast(p=0.5),
                      A.RandomBrightnessContrast(p=0.5),
                  ])
                  ), batch_size=10, shuffle=True, num_workers=1, pin_memory=False
)

val_loader = torch.utils.data.DataLoader(
    XunFeiDataset(train_path.values[-500:], train_label.values[-500:],
                  A.Compose([
                      # A.Resize(300, 300),
                      A.RandomCrop(130, 450),
                      # A.HorizontalFlip(p=0.5),
                      # A.RandomContrast(p=0.5),
                  ])
                  ), batch_size=2, shuffle=False, num_workers=1, pin_memory=False
)

test_loader = torch.utils.data.DataLoader(
    XunFeiDataset(test_path, [0] * len(test_path),
                  A.Compose([
                      # A.Resize(300, 300),
                      A.RandomCrop(130, 450),
                      # A.HorizontalFlip(p=0.5),
                      # A.RandomContrast(p=0.5),
                  ])
                  ), batch_size=2, shuffle=False, num_workers=1, pin_memory=False
)


def train(train_loader, model, criterion, optimizer):
    model.train()
    train_loss = 0.0
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 20 == 0:
            print(loss.item())

        train_loss += loss.item()
    return train_loss/len(train_loader.dataset), model


def validate(val_loader, model, criterion):
    model.eval()

    val_acc = 0.0

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            val_acc += (output.argmax(1) == target).sum().item()

    return val_acc / len(val_loader.dataset)




def aaa():
    model = XunFeiNet()
    model = model.to('cuda')
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.001)
    loss = list()
    acc = list()
    for _ in range(2):
        train_loss, model = train(train_loader, model, criterion, optimizer)
        val_acc = validate(val_loader, model, criterion)
        loss.append(train_loss)
        acc.append(val_acc)
        print(train_loss, val_acc)
    n = [i+1 for i in range(2)]
    plt.plot(n, loss)
    plt.figure(1)
    plt.xlabel("n")
    plt.ylabel("loss")
    plt.savefig('./loss.png')
    plt.plot(n, acc)
    plt.figure(2)
    plt.xlabel("n")
    plt.ylabel("acc")
    plt.savefig('/acc.png')


if __name__ == '__main__':
    import fire
    fire.Fire()