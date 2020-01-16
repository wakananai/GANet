import numpy as np
import argparse
import cv2
import time
from util import readPFM, cal_avgerr

parser = argparse.ArgumentParser(description='Disparity Estimation')
parser.add_argument('--pred', default='./output/TL0.pfm', type=str, help='Predicted disparity')
parser.add_argument('--gt', default='./custom_data/Synthetic/TLD0.pfm', type=str, help='Ground truth disparity')


def main():
    args = parser.parse_args()

    print('Compute disparity error for %s' % args.pred)
    pred = readPFM(args.pred)
    gt = readPFM(args.gt)
    err = cal_avgerr(gt, pred)
    print(err)


if __name__ == '__main__':
    main()
