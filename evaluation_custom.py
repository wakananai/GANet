from __future__ import print_function
import argparse
import sys
# import shutil
import os
import os.path as op
import re
from struct import unpack
# from math import log10

# from dataloader.data import get_test_set
import skimage
import skimage.io
import skimage.transform
from PIL import Image
import numpy as np
import torch
# import torch.nn as nn
import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.optim as optim
from torch.autograd import Variable
# from torch.utils.data import DataLoader
import cv2

from models.GANet_deep import GANet
import opt


def build_env():
    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    print('===> Building model')
    model = GANet(opt.max_disp)

    if cuda:
        model = model.cuda()

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            model.load_state_dict(checkpoint['state_dict'], strict=False)

        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    return model, cuda


def readPFM(file):
    with open(file, "rb") as f:
        # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        type = f.readline().decode('latin-1')
        if "PF" in type:
            channels = 3
        elif "Pf" in type:
            channels = 1
        else:
            sys.exit(1)
        # Line 2: width height
        line = f.readline().decode('latin-1')
        width, height = re.findall(r'\d+', line)
        width = int(width)
        height = int(height)

        # Line 3: +ve number means big endian, negative means little endian
        line = f.readline().decode('latin-1')
        BigEndian = True
        if "-" in line:
            BigEndian = False
        # Slurp all binary data
        samples = width * height * channels
        buffer = f.read(samples * 4)
        # Unpack floats with appropriate endianness
        if BigEndian:
            fmt = ">"
        else:
            fmt = "<"
        fmt = fmt + str(samples) + "f"
        img = unpack(fmt, buffer)
        img = np.reshape(img, (height, width))
        img = np.flipud(img)
        # cv2.imwrite('./result/xxxx.png', img)
    return img, height, width


def test_transform(temp_data, crop_height, crop_width):
    _, h, w = np.shape(temp_data)

    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([6, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp
    else:
        raise ValueError
        start_x = int((w - crop_width) / 2)
        start_y = int((h - crop_height) / 2)
        # print(start_x, start_y)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]

    left, right = split_left_right(temp_data, crop_height, crop_width)
    return left, right, h, w


def split_left_right(temp_data, crop_height, crop_width):
    left = np.ones([1, 3, crop_height, crop_width], 'float32')
    left[0, :, :, :] = temp_data[0: 3, :, :]
    right = np.ones([1, 3, crop_height, crop_width], 'float32')
    right[0, :, :, :] = temp_data[3: 6, :, :]
    return torch.from_numpy(left).float(), torch.from_numpy(right).float()


def load_data(leftarray, rightarray):
    left = Image.fromarray(leftarray)
    right = Image.fromarray(rightarray)
    # temp crop size
    if opt.policy == 'directly_resize':
        left = left.resize((opt.crop_width, opt.crop_height))
        right = right.resize((opt.crop_width, opt.crop_height))
    elif opt.policy == 'resize_to_slide':
        out_width = opt.crop_width
        height, width = np.shape(left)[:2]
        out_height = int(height * (out_width / width))
        left = left.resize((out_width, out_height))
        right = right.resize((out_width, out_height))

    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([6, height, width], 'float32')
    left = np.asarray(left)
    right = np.asarray(right)
    r = left[:, :, 0]
    g = left[:, :, 1]
    b = left[:, :, 2]
    temp_data[0, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[1, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[2, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    r = right[:, :, 0]
    g = right[:, :, 1]
    b = right[:, :, 2]
    # r,g,b,_ = right.split()
    temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    return temp_data


def slide_h(img, window_size: int, overlapping_h: int):
    c, h, w = np.shape(img)
    step_size = window_size - overlapping_h
    total_steps = int(np.ceil((h - window_size) / step_size))
    required_h = int(total_steps * step_size + window_size)

    padded_h = required_h - h
    img = np.pad(img, ((0, 0), (0, padded_h), (0, 0)), mode='constant', constant_values=0)
    imgs = [
        img[:, i * step_size: i * step_size + window_size, :]
        for i in range(total_steps + 1)
    ]
    return imgs, padded_h


def deslide_h(imgs, window_size: int, overlapping_h: int, padded_h: int, ensemble='overlap'):
    total_steps, h, w = np.shape(imgs)
    step_size = window_size - overlapping_h

    total_h = (total_steps - 1) * step_size + window_size
    out = np.zeros([total_h, w])
    for i in range(total_steps):
        out[i * step_size: i * step_size + window_size, :] = imgs[i]
    return out[: total_h - padded_h, :]


def test(leftarray, rightarray, model, cuda):
    datas = load_data(leftarray, rightarray)

    if opt.policy == 'resize_to_slide':
        slided_datas, padded_h = slide_h(datas, opt.crop_height, opt.sliding_window_overlap)
        input1, input2 = zip(*[
            split_left_right(window, opt.crop_height, opt.crop_width)
            for window in slided_datas
        ])
        input1 = torch.stack(input1, axis=0).squeeze(axis=1)
        input2 = torch.stack(input2, axis=0).squeeze(axis=1)
    else:
        input1, input2, height, width = test_transform(datas, opt.crop_height, opt.crop_width)
        input1 = Variable(input1, requires_grad=False)
        input2 = Variable(input2, requires_grad=False)

    model.eval()
    if cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()
    with torch.no_grad():
        prediction = model(input1, input2)

    output = prediction.cpu().detach().numpy()
    if opt.policy == 'resize_to_slide':
        output = deslide_h(output, opt.crop_height, opt.sliding_window_overlap, padded_h)
    else:
        if height <= opt.crop_height and width <= opt.crop_width:
            output = output[0, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
        else:
            raise ValueError
            output = output[0, :, :]

    return output


def evaluate(leftarray, rightarray):
    model, cuda = build_env()
    prediction = test(leftarray, rightarray, model, cuda)
    # Resize and rescale predicted disparity map
    our_height, our_width, ch = leftarray.shape

    print(f"Pridction shape: {prediction.shape}")
    if opt.policy == 'directly_resize':
        prediction = cv2.resize(prediction, (our_width, our_height))
        prediction /= (opt.crop_width / our_width)
        print(f"Pridction shape after resize back: {prediction.shape}")
    elif opt.policy == 'resize_to_slide':
        h, w = prediction.shape
        assert abs(h / our_height - w / our_width) < 1e-5
        prediction = cv2.resize(prediction, (our_width, our_height))
        prediction /= (h / our_height)
    return prediction


if __name__ == '__main__':
    lefta = cv2.imread('/tmp2/r07922076/CV_final/color/Real/TL0.png')
    righta = cv2.imread('/tmp2/r07922076/CV_final/color/Real/TR0.png')
    prediction =  eval(lefta, righta)

    pred_out_path = './out.png'
    print(f"Saving prediction to {pred_out_path}")
    normal = (prediction -  prediction.min()) / (prediction.max() - prediction.min())
    skimage.io.imsave(pred_out_path, (normal * 255).astype('uint16'))