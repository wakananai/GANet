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


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch GANet Example')
    parser.add_argument('--crop_height', type=int, help="crop height", default=240)
    parser.add_argument('--crop_width', type=int, help="crop width", default=624)
    parser.add_argument('--max_disp', type=int, default=192, help="max disp")
    parser.add_argument('--resume', type=str, default='../sceneflow_epoch_10.pth', help="resume from saved model")
    parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
    parser.add_argument('--data_path', type=str, help="data root", default='../CV_finalproject/data')
    parser.add_argument('--save_path', type=str, default='./result/', help="location to save result")
    parser.add_argument('--threshold', type=float, default=3.0, help="threshold of error rates")
    parser.add_argument('--multi_gpu', type=int, default=0, help="multi_gpu choice")
    parser.add_argument(
        '--policy', type=str, default='cropping',
        choices=['cropping', 'directly_resize', 'resize_to_slide'])
    parser.add_argument('--sliding_window_overlap', type=int, default=30)

    opt = parser.parse_args()
    return opt

# print('===> Loading datasets')
# test_set = get_test_set(opt.data_path, opt.test_list, [opt.crop_height, opt.crop_width],
#                         false, opt.kitti, opt.kitti2015)
# testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads,
#                                  batch_size=opt.testBatchSize, shuffle=False)


def build_env(opt):
    cuda = opt.cuda
    # cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

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


def load_data(leftname, rightname, opt):
    left = Image.open(leftname)
    right = Image.open(rightname)
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


def test(opt, leftname, rightname, model, cuda):
    datas = load_data(leftname, rightname, opt)

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


def main():
    opt = parse_args()
    os.makedirs(op.join(opt.save_path, 'pred'), exist_ok=True)
    os.makedirs(op.join(opt.save_path, 'gt'), exist_ok=True)
    print(opt)

    model, cuda = build_env(opt)

    dataset_len = 10
    avg_error = 0
    avg_rate = 0

    for index in range(dataset_len):
        leftname = op.join(opt.data_path, 'Synthetic', ('TL%d.png' % index))
        rightname = op.join(opt.data_path, 'Synthetic', ('TR%d.png' % index))
        gt_disp_path = op.join(opt.data_path, 'Synthetic', ('TLD%d.pfm' % index))

        pred_out_path = op.join(opt.save_path, 'pred', f'{str(index)}.png')
        gt_viewable_path = op.join(opt.save_path, 'gt', f'{str(index)}.png')

        # Read ground truth PFM and save it as png
        disp, our_height, our_width = readPFM(gt_disp_path)
        print(f"Saving ground truth image (as a viewable file) into: {gt_viewable_path}")
        skimage.io.imsave(gt_viewable_path, (disp * 256).astype('uint16'))
        print(f"Ground truth max disparity: {disp.max(): .4f}")

        prediction = test(opt, leftname, rightname, model, cuda)

        # Resize and rescale predicted disparity map
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

        print(f"Saving prediction to {pred_out_path}")
        skimage.io.imsave(pred_out_path, (prediction * 256).astype('uint16'))

        error = np.mean(np.abs(prediction - disp))
        rate = np.mean(np.abs(prediction - disp) > opt.threshold)
        avg_error += error
        avg_rate += rate
        print("===> Frame {}: ".format(index) + " ==> EPE Error: {:.4f}, Error Rate: {:.4f}".format(error, rate))
    avg_error = avg_error / dataset_len
    avg_rate = avg_rate / dataset_len
    print("===> Total {} Frames ==> AVG EPE Error: {:.4f}, AVG Error Rate: {:.4f}".format(
        dataset_len, avg_error, avg_rate))


if __name__ == "__main__":
    main()
