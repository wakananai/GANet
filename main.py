import numpy as np
import argparse
import cv2
import time
from util import writePFM
from evaluation_custom import evaluate
from colorization_extract.colorize_func import gray2color

parser = argparse.ArgumentParser(description='Disparity Estimation')
parser.add_argument('--input-left', default='./data/Synthetic/TL0.png', type=str, help='input left image')
parser.add_argument('--input-right', default='./data/Synthetic/TR0.png', type=str, help='input right image')
parser.add_argument('--output', default='./TL0.pfm', type=str, help='left disparity map')


def isGray(Il, Ir):
    return ((Il[:,:,0] == Il[:,:,1]) & (Il[:,:,1] == Il[:,:,2])).all() 

# You can modify the function interface as you like
def computeDisp(Il, Ir):
    print(Il.shape)
    if not isGray(Il, Ir):
        # transfer to RGB
        Il = Il[:,:,::-1]
        Ir = Ir[:,:,::-1]
        # This is rgb image
        # Apply GANet
        disp = evaluate(Il, Ir)
    else:
        # This is gray scale image(input 3 channel R==G==B)
        # Apply colorization
        Il = gray2color(Il)
        Ir = gray2color(Ir)

        # add padding to avoid negtive disparity
        h,w,c = Il.shape

        Il_pad = np.zeros((h,w+30,3), np.uint8)
        Ir_pad = np.zeros((h,w+30,3), np.uint8)

        Il_pad[:, 30:, :] = Il
        Ir_pad[:, :-30, :] = Ir

        # Apply GANet
        # disp = evaluate(Il, Ir)
        disp = evaluate(Il_pad, Ir_pad)
        disp = disp[:,30:]
        value_95th = np.percentile(disp,95)
        value_5th = np.percentile(disp,5)
        disp = np.clip(disp, value_5th, value_95th)
        disp -= disp.min()


    return disp.astype(np.float32)


def main():
    args = parser.parse_args()

    print(args.output)
    print('Compute disparity for %s' % args.input_left)
    img_left = cv2.imread(args.input_left)
    img_right = cv2.imread(args.input_right)
    tic = time.time()
    disp = computeDisp(img_left, img_right)
    toc = time.time()
    writePFM(args.output, disp)
    print('Elapsed time: %f sec.' % (toc - tic))


if __name__ == '__main__':
    main()
