import glob
import os.path as osp
import random
import numpy as np
import json
from PIL import Image
# from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
    
import torchvision.models as models

import torchvision
from torchvision import models, transforms,datasets
from torch.utils.data import Dataset, TensorDataset
import os, sys, time, datetime
import torchvision
import argparse

from models.resnet18 import ResNet18
from models.wide_resnet import WideResNet
from libs.smooth_cross_entropy import smooth_crossentropy
from libs.auto_augment import AutoAugment, Cutout, str2bool
import wandb
torch.manual_seed(0)

# 乱数のシードを設定
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


parser = argparse.ArgumentParser(description='This script is pytorch template')
# parser.add_argument('--save_path',default=".")    
parser.add_argument('--data',default="cifar10")
parser.add_argument('--batch_size',default=128)
parser.add_argument('--lr',default=0.01)
parser.add_argument('--epochs',default=1)
parser.add_argument('--cuda',default=0)
parser.add_argument('--model',default="resnet18",choices=["resnet18", "wide-resnet28-10"])
parser.add_argument('--optimizer',default="SGD",choices=["SGD", "SAM"])
parser.add_argument('--auto_augment', default=False, type=str)
parser.add_argument('--debug_mode', default=False, type=str)
args = parser.parse_args()
# print(args.accumulate(args.integers))

print(args.auto_augment)
augment = "none"
if "autoaugment" == args.auto_augment:
    augment = args.auto_augment
if "base" == args.auto_augment:
    augment = args.auto_augment

if str(args.debug_mode) == "True":
    debug = "online"
else:
    debug = "offline"

run_name = args.model+"-"+args.data+"-"+str(args.epochs)

wandb.init(
    project = "template",
    name = run_name,
    mode = debug,
    group = "hogehoge",

    config={
        "batch_size": int(args.batch_size),
        "learn_rate": float(args.lr),
        "epochs": int(args.epochs) ,
        "data": str(args.data),
        "model": str(args.model),
        "optimizer": str(args.optimizer),
        "augmentation":str(augment)
        })

config = wandb.config


############## parameter ###################
BATHCSIZE = wandb.config.batch_size
EPOCHS = wandb.config.epochs
LEANRATE = wandb.config.learn_rate
model_name = str(args.model)
optimizer_name = str(args.optimizer)

device = 'cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu'
print(device)
############## create dataset###################
transform = []
if "autoaugment" == augment:
    transform.append(AutoAugment())
if "base" == augment:
    transform.append(transforms.RandomRotation(degrees=30))
    transform.append(transforms.RandomHorizontalFlip(p=0.5))
    
transform.append( transforms.ToTensor() )
transform.append( transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) )
transform_train = transforms.Compose(transform)

transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

if str(args.data) =="cifar10":
    train_set = torchvision.datasets.CIFAR10(root="../datasets", train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATHCSIZE, shuffle=True, num_workers=4)
    test_set = torchvision.datasets.CIFAR10(root="../datasets", train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=4)
    num_class=10
if str(args.data) =="cifar100":
    train_set = torchvision.datasets.CIFAR100(root="../datasets", train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATHCSIZE, shuffle=True, num_workers=4)
    test_set = torchvision.datasets.CIFAR100(root="../datasets", train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=4)
    num_class=100
if str(args.data) =="fashion_mnist":
    train_set = torchvision.datasets.FashionMNIST(root="../datasets", train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATHCSIZE, shuffle=True, num_workers=4)
    test_set = torchvision.datasets.FashionMNIST(root="/mnt/ds/ryo", train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=4)
    num_class = 10

############## parameter ###################
if str(args.model) =="resnet18":
    model = ResNet18(num_class).to(device)
if str(args.model) =="wide-resnet28-10":
    model = WideResNet(28, 10, 0, in_channels=3, labels=num_class).to(device)


# optimizing
if str(args.optimizer) =="SGD":
    optimizer = optim.SGD(model.parameters(), lr=LEANRATE, momentum=0)
if str(args.optimizer) =="SAM":
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=0.05,lr=LEANRATE, momentum=0,weight_decay=0.0005)

criterion =  nn.CrossEntropyLoss()

# training
print('training start ...')

# initialize list for plot graph after training
train_loss_list, train_acc_list= [], []
test_loss_list, test_acc_list= [], []
start_time = time.time()
for epoch in range(EPOCHS):
    # initialize each epoch
    train_loss, train_acc= 0, 0
    test_loss, test_acc= 0, 0

    # ======== train mode ======
    model.train()
    for i, (images, labels) in enumerate(train_loader):  # ミニバッチ回数実行
        # 画像をdeviceへ転送
        images, labels=images.to(device), labels.to(device)
        optimizer.zero_grad()  # 勾配リセット
 
        # forward 
        outputs = model(images)  # 順伝播の計算

        #SAM backprop
        if str(args.optimizer) =="SAM":
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            loss = criterion(model(images),labels)
            loss.backward()
            optimizer.second_step()

        #SGD backprop
        if str(args.optimizer) =="SGD":
            loss = criterion(outputs,labels) # output loss
            loss.backward()  # 逆伝播の計算
            optimizer.step()  # 重みの更新

        acc = (outputs.max(1)[1] == labels).sum()  # 予測とラベルが合っている数の合計

        train_loss += loss.item()  # train_loss に結果を蓄積
        train_acc += acc.item()  # train_acc に結果を蓄積

    avg_train_loss = train_loss / len(train_loader)  # lossの平均を計算
    avg_train_acc = train_acc / len(train_loader.dataset)   # accの平均を計算

    # ======== test mode =====
    model.eval()
    with torch.no_grad():
        total = 0
        test_acc = 0
        for images, labels in test_loader:
            images, labels=images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs,labels)

            test_acc += (outputs.max(1)[1] == labels).sum().item()
            total += labels.size(0)
            test_loss += loss.item()  # train_loss に結果を蓄積
    avg_test_loss = test_loss / len(test_loader)  # lossの平均を計算
    avg_test_acc = test_acc / len(test_loader.dataset)  # accの平均を計算

    # print log
    print('Epoch [{}/{}], train_loss: {loss:.8f}, train_acc: {train_acc:.4f}'.format(epoch+1, EPOCHS, loss=avg_train_loss, train_acc=avg_train_acc))
    print('Epoch [{}/{}], test_loss: {loss:.8f}, test_acc: {train_acc:.4f}'.format(epoch+1, EPOCHS, loss=avg_test_loss, train_acc=avg_test_acc))
    
    # append list for polt graph after training
    train_loss_list.append(avg_train_loss)
    train_acc_list.append(avg_train_acc)
    test_loss_list.append(avg_test_loss)
    test_acc_list.append(avg_test_acc)
    wandb.log({"epoch": epoch+1, "train accuracy": avg_train_acc})
    wandb.log({"epoch": epoch+1, "train accuracy": avg_train_acc})
    wandb.log({"epoch": epoch+1, "test accuracy": avg_test_acc})
    wandb.log({"epoch": epoch+1, "train loss": avg_train_loss})
    wandb.log({"epoch": epoch+1, "test loss": avg_test_loss})

end_time = time.time()
print('elapsed time: {:.4f}'.format(end_time-start_time))

torch.save(model.state_dict(),"result/model_weight/"+run_name+'.pth')
wandb.save("result/model_weight/"+run_name+'.pth')

