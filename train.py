#-*- coding:utf-8 -*-

from __future__ import unicode_literals, print_function, division
from skimage import io, transform, filters
from torchvision.models import resnet18
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset
from pprint import pprint
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from network import MobileCRNN
from dataset import *

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epoch', default=100, type=int, help='Epoch')
parser.add_argument('-b', '--batch_size', default=64, type=int, help='Batch size')
parser.add_argument('-l', '--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('-L', '--lamb', default=0.0001, type=float, help='N2 normalization')
parser.add_argument('-o', '--optim', default='adam', type=str, help='Epoch')

args = parser.parse_args()

chars = "*0123456789.-" # * for blank and padding id is also blank
char2id = {char:i for i, char in enumerate(chars)}
id2char = {i: char for i, char in enumerate(chars)}

epochs = args.epoch
batch_size = args.batch_size
lr = args.lr
lamb = args.lamb

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = MobileCRNN(hidden_dim=128, vocab_size=len(chars)).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr) if args.optim == 'adam' else optim.SGD(model.parameters(), lr=lr, momentum=0.09)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)

train_dataset = DigitDataset(picklefile='datas_1_7_3channel.pkl', transform=transforms.Compose([ ReScale(), ToTensor()]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# for batch in train_loader:
#     print(batch['image'].shape, batch['text'].shape, batch['length'].shape)
#     break

train_losses = []

for epoch in tqdm(range(1, epochs+1)):
    total_loss = 0
    for i, sample in enumerate(train_loader):
        batch_size = sample['image'].shape[0]
        optimizer.zero_grad()
        x = Variable(sample['image']).to(device)
        y = Variable(sample['text']).view(batch_size, -1).to(device)
        l = Variable(sample['length']).to(device)
        il = Variable(torch.full(size=(batch_size,), fill_value=10, dtype=torch.long)).to(device)

        o = model(x) #.permute(2, 0, 1)
        loss = criterion(o, y, il, l)
        #print(loss)
        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss += lamb * l2_reg
        loss.backward()
        optimizer.step()
        #print(loss)
        total_loss += loss.detach()
        losses.append(total_loss)
    if epoch % 1==0:
       print("Epoch [%d/%d] loss=%.3f" % (epoch, epochs, total_loss))

torch.save(model.state_dict(), 'mobilecrnnv2_epoch{}.pt'.format(epochs))
plt.plot(losses)
plt.show()
