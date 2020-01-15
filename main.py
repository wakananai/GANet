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
        disp = evaluate(Il, Ir).astype(np.int32)
    else:
        # This is gray scale image(input 3 channel R==G==B)
        # Apply colorization
        Il = gray2color(Il)
        Ir = gray2color(Ir)
        # Apply GANet
        disp = evaluate(Il, Ir).astype(np.int32)

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
