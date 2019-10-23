#-*- coding:utf-8 -*-

from skimage import io, transform, filters
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import cv2
import os

chars = "*0123456789.-"
char2id = {char:i for i, char in enumerate(chars)}
id2char = {i: char for i, char in enumerate(chars)}


class DigitDataset(Dataset):
    """
     Load from pickle file
    """
    def __init__(self, picklefile='datas_1_7_3channel.pkl', height=32, maxlen=10, maxwid=120, transform=None, rotate90=False):
        self.picklefile = picklefile
        self.datas = self.make_datas()
        self.height = height
        self.maxlen = maxlen
        self.maxwid = maxwid
        self.transform = transform
        self.rotate90 = rotate90

        self.blur_rate = 0.5
        self.change_channel_rate = 0.5
        self.shift_rate = 0.5
        self.rotate_rate = 0.2
        self.bend_rate = 0.3
        self.head_pad_rate = 0.3

        self.blur_size = 3
        self.rotate_angle = 10
        self.bend_angle = 7

    def make_datas(self):
        with open(self.picklefile, 'rb') as f:
            datas = pickle.load(f)
        return datas

    def __len__(self):
        return len(self.datas)


    def pad_img(self, img, head=False):
        maxwid = self.maxwid
        h, w = img.shape[:2]
        if w > maxwid:
            img = cv2.resize(img, (120, h))
        h, w = img.shape[:2]
        mn = np.median(img)
        assert h==32
        pad = np.full((32, maxwid-w, 3), mn)
        if not head:
            return np.concatenate([img, pad], 1).clip(0, 255).astype(np.uint8)
        else:
            return np.concatenate([pad, img], 1).clip(0, 255).astype(np.uint8)

    def change_channel(self, img):
        #RGB -> BGR, BRG
        bgr = np.random.choice([True, False])
        indexes = [2, 1, 0] if bgr else [2, 0, 1]
        return img[:, :, indexes]

    def shift(self, img, value, length, right=True):
        length = len(value)
        w = img.shape[1]
        dw = w // length
        if right:
            img = img[:, 0:w-dw, ::]
            value = value[:-1]
        else:
            img = img[:, dw:w, ::]
            value = value[1:]
        length = len(value)
        return img, value, length

    def rotate(self, img, scale=1.0):
        maxangle = self.rotate_angle
        h, w = img.shape[:2]
        angle = np.random.randint(-maxangle, maxangle)
        center = (int(w/2), int(h/2))
        trans = cv2.getRotationMatrix2D(center, angle , scale)
        return cv2.warpAffine(img, trans, (w,h))

    def bend(self, img, scale=1.0):
        maxangle = self.bend_angle

        h, w = img.shape[:2]
        angle = np.random.randint(0, maxangle)

        center = (int(w/2), int(h/2))
        trans = cv2.getRotationMatrix2D(center, -angle , scale)
        angle /= 57.3

        left_down, right_down = [h, 0], [h, w]
        left_up, right_up = [0, 0], [0, w]
        perspective1 = np.float32([left_down,right_down,right_up,left_up])
        perspective2 = np.float32([left_down,[h ,int(w-h*np.tan(angle))], right_up,[0, int(h*np.tan(angle))]])
        #perspective2 = np.float32([[h, -int(h*np.tan(angle))], right_down, [0 , int(w+h*np.tan(angle))], left_up])
        psp_matrix = cv2.getPerspectiveTransform(perspective1,perspective2)
        psp =  cv2.warpPerspective(img, psp_matrix,(w, h))

        return cv2.warpAffine(psp, trans, (w,h))


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.datas[idx]
        input_img = data['image']
        value = str(data['value'])
        length = int(data['length'])

        h, w = input_img.shape[:2]
        if w > self.maxwid:
           input_img = cv2.resize(input_img, (self.maxwid, self.height))

        if np.random.rand() < self.blur_rate:
            input_img = cv2.blur(input_img, (self.blur_size, self.blur_size))
        if np.random.rand() < self.change_channel_rate:
            input_img = self.change_channel(input_img)
        if np.random.rand() < self.shift_rate:
            input_img, value, length = self.shift(input_img, value, length)
        if np.random.rand() < self.rotate_rate:
            input_img = self.rotate(input_img)
        if np.random.rand() < self.bend_rate:
            input_img = self.bend(input_img)
        if np.random.rand() < self.head_pad_rate:
            input_img = self.pad_img(input_img, head=True)
            value = '*' * (self.maxlen-length) + value
        else:
            input_img = self.pad_img(input_img, head=False)
            value += '*' * (self.maxlen-length)

        #print(value, length, len(value))
        if self.rotate90:
            input_img = cv2.rotate(input_img, cv2.ROTATE_90_CLOCKWISE)
        sample = {'image':input_img, 'text': value, 'length':length}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

class ReScale(object):
    def __call__(self, sample):
        image = sample['image'].astype(np.float32)
        maxpix = np.max(image)
        if maxpix < 1.1:
           return sample
        else:
           image /= 255.
           sample['image'] = image
           return sample

class ToTensor(object):
    def __call__(self, sample, maxlen = 10):
        image, value, length = sample['image'], sample['text'], sample['length']
        if len(image.shape) > 2:
            image = image.transpose((2, 0, 1))
        ids = [char2id[char] for char in value]
        if len(ids) < maxlen:
            ids += [char2id['*']] * (maxlen-len(ids))
        return {'image': torch.from_numpy(image).type(torch.FloatTensor),
                'text': torch.Tensor([ids]).type(torch.LongTensor),
                'length': torch.Tensor([length]).type(torch.LongTensor)
                }

if __name__ == '__main__':
    generator = DigitDataset(transform=transforms.Compose([ ReScale(), ToTensor()]), rotate90=True )
    sample = generator[0]

    print(sample['image'].shape, sample['text'].shape)
    print(sample['text'], sample['length'])
    from network import MobileCRNN
    model = MobileCRNN(128, len(chars))
    criterion = torch.nn.CTCLoss(blank=0)
    x = sample['image'].unsqueeze(0)
    y = sample['text']
    l = sample['length']
    o = model(x) #.permute(1, 0, 2)
    input_length = torch.full(size=(1,), fill_value=10, dtype=torch.long)
    print(o.shape, y.shape, input_length.shape, l.shape)
    loss = criterion(o, y, input_length, l)
    print(loss)
