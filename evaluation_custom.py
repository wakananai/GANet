from __future__ import print_function
import argparse
import skimage
import skimage.io
import skimage.transform
from PIL import Image
from math import log10
import sys
import shutil
import os
import re
from struct import unpack
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.GANet_deep import GANet

#from dataloader.data import get_test_set
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='PyTorch GANet Example')
parser.add_argument('--crop_height', type=int, required=True, help="crop height")
parser.add_argument('--crop_width', type=int, required=True, help="crop width")
parser.add_argument('--max_disp', type=int, default=192, help="max disp")
parser.add_argument('--resume', type=str, default='', help="resume from saved model")
parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
parser.add_argument('--data_path', type=str, required=True, help="data root")
parser.add_argument('--save_path', type=str, default='./result/', help="location to save result")
parser.add_argument('--threshold', type=float, default=3.0, help="threshold of error rates")
parser.add_argument('--multi_gpu', type=int, default=0, help="multi_gpu choice")

opt = parser.parse_args()


print(opt)

cuda = opt.cuda
#cuda = True
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

#print('===> Loading datasets')
#test_set = get_test_set(opt.data_path, opt.test_list, [opt.crop_height, opt.crop_width], false, opt.kitti, opt.kitti2015)
#testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
model = GANet(opt.max_disp)

if cuda:
    model = torch.nn.DataParallel(model).cuda()

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))


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
        width, height = re.findall('\d+', line)
        width = int(width)
        height = int(height)

            # Line 3: +ve number means big endian, negative means little endian
        line = f.readline().decode('latin-1')
        BigEndian = True
        if "-" in line:
            BigEndian = False
        # Slurp all binary data
        samples = width * height * channels;
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
        # print(img.max())
    return img, height, width

def test_transform(temp_data, crop_height, crop_width):
    _, h, w=np.shape(temp_data)

    # print(temp_data.shape)
    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([6, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp
    else:
        start_x = int((w - crop_width) / 2)
        start_y = int((h - crop_height) / 2)
        # print(start_x, start_y)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
    left = np.ones([1, 3,crop_height,crop_width],'float32')
    left[0, :, :, :] = temp_data[0: 3, :, :]
    right = np.ones([1, 3, crop_height, crop_width], 'float32')
    right[0, :, :, :] = temp_data[3: 6, :, :]
    return torch.from_numpy(left).float(), torch.from_numpy(right).float(), h, w

def load_data(leftname, rightname):
    left = Image.open(leftname)
    right = Image.open(rightname)
    # temp crop size
    left = left.resize((624,240))
    right = right.resize((624,240))

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
    #r,g,b,_ = right.split()
    temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    return temp_data

def test(leftname, rightname, savename):
    input1, input2, height, width = test_transform(load_data(leftname, rightname), opt.crop_height, opt.crop_width)
    input1 = Variable(input1, requires_grad = False)
    input2 = Variable(input2, requires_grad = False)

    model.eval()
    if cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()
    with torch.no_grad():
        prediction = model(input1, input2)

    temp = prediction.cpu()
    temp = temp.detach().numpy()
    if height <= opt.crop_height and width <= opt.crop_width:
        temp = temp[0, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
    else:
        temp = temp[0, :, :]

    temp = cv2.resize(temp, (384,512))
    skimage.io.imsave(savename, (temp * 256).astype('uint16'))

    return temp


if __name__ == "__main__":
    file_path = opt.data_path
    dataset_len = 10
    avg_error = 0
    avg_rate = 0

    for index in range(dataset_len):
        leftname = opt.data_path + 'Synthetic/' + ('TL%d.png' %index)
        rightname = opt.data_path + 'Synthetic/' + ('TR%d.png' %index)
        dispname = opt.data_path + 'Synthetic/' + ('TLD%d.pfm' %index)

        savename = opt.save_path + str(index) + '.png'
        disp, height, width = readPFM(dispname)

        prediction = test(leftname, rightname, savename)
        print(prediction.shape)
        prediction = cv2.resize(prediction, (384,512))

        mask = np.logical_and(disp >= 0.001, disp <= opt.max_disp)

        error = np.mean(np.abs(prediction[mask] - disp[mask]))
        rate = np.sum(np.abs(prediction[mask] - disp[mask]) > opt.threshold) / np.sum(mask)
        avg_error += error
        avg_rate += rate
        print("===> Frame {}: ".format(index) + " ==> EPE Error: {:.4f}, Error Rate: {:.4f}".format(error, rate))
    avg_error = avg_error / dataset_len
    avg_rate = avg_rate / dataset_len
    print("===> Total {} Frames ==> AVG EPE Error: {:.4f}, AVG Error Rate: {:.4f}".format(dataset_len, avg_error, avg_rate))

