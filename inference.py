#-*- coding:utf-8 -*-

from network import *
from skimage import filters
from glob import glob
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import pickle
import cv2
import os

chars = "*0123456789.-"
id2char = {i: char for i, char in enumerate(chars)}

def decode_output(output_str):
    chars_ = ' '
    for c in output_str:
        if c != chars_[-1]:
            chars_ += c
    chars_ = chars_[1:]
    chars_ = chars_.replace('*', '')
    chars_ = chars_.replace('+', '')
    return chars_

def main(PATH):
    model = MobileCRNN(128, len(chars))

    model.load_state_dict(torch.load(PATH, map_location='cpu'))
    model.eval()

    imgfiles = glob(os.path.join('./imgs', '*'))

    num = 24
    with torch.no_grad():
        plt.figure(figsize=(10, 3))
        for i, imgfile in enumerate(imgfiles):
            img = cv2.imread(imgfile)[:, :, ::-1]
            h, w = img.shape[:2]
            h_, w_ = 32, int(32 / h * w)
            img = cv2.resize(img, (w_, 32))
            im = img.transpose((2, 0, 1)) / 255.
            input_tensor = torch.from_numpy(im).unsqueeze(0)
            input_tensor = input_tensor.type(torch.FloatTensor)
            output = model(input_tensor)
            #print(output.shape)
            _, topi = output.topk(1)
            ids = topi.view(-1,).detach().numpy()
            chars_ = ''.join([id2char[ID] for ID in ids])
            decoded = decode_output(chars_)
            #chars_ = chars_.replace('*', '')

            plt.subplot(4, 6, i+1)
            plt.title(decoded)
            plt.imshow(img)
            if i > num-2:
                break

    # print(img.shape)
    #plt.imshow(img, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main('models/mobilecrnn_epoch100.pt')
