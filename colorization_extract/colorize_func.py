import numpy as np
import os
import skimage.color as color
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation as sni
os.environ["GLOG_minloglevel"] = "1"
import caffe
from colorization_extract import config
import cv2

# need to download model via
# ./models/fetch_release_models.sh

def gray2color(img_in_array, gpu_id=0):

    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)

    # Select desired model
    net = caffe.Net(config.prototxt, config.caffemodel, caffe.TEST)

    (H_in,W_in) = net.blobs['data_l'].data.shape[2:] # get input shape
    (H_out,W_out) = net.blobs['class8_ab'].data.shape[2:] # get output shape

    pts_in_hull = np.load(config.pts_in_hull_numpy) # load cluster centers
    net.params['class8_ab'][0].data[:,:,0,0] = pts_in_hull.transpose((1,0)) # populate cluster centers as 1x1 convolution kernel
    # print 'Annealed-Mean Parameters populated'

    # load the original image
    img_rgb = (img_in_array.astype(np.float32) / 255.0)[:,:,::-1]
    # img_rgb = caffe.io.load_image(img_in)


    img_lab = color.rgb2lab(img_rgb) # convert image to lab color space
    img_l = img_lab[:,:,0] # pull out L channel
    (H_orig,W_orig) = img_rgb.shape[:2] # original image size

    # create grayscale version of image (just for displaying)
    img_lab_bw = img_lab.copy()
    img_lab_bw[:,:,1:] = 0
    img_rgb_bw = color.lab2rgb(img_lab_bw)

    # resize image to network input size
    img_rs = caffe.io.resize_image(img_rgb,(H_in,W_in)) # resize image to network input size
    img_lab_rs = color.rgb2lab(img_rs)
    img_l_rs = img_lab_rs[:,:,0]

    net.blobs['data_l'].data[0,0,:,:] = img_l_rs-50 # subtract 50 for mean-centering
    net.forward() # run network

    ab_dec = net.blobs['class8_ab'].data[0,:,:,:].transpose((1,2,0)) # this is our result
    ab_dec_us = sni.zoom(ab_dec,(1.*H_orig/H_out,1.*W_orig/W_out,1)) # upsample to match size of original image L
    img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2) # concatenate with original image L
    img_rgb_out = (255*np.clip(color.lab2rgb(img_lab_out),0,1)).astype('uint8') # convert back to rgb

    return img_rgb_out


if __name__ == '__main__':
    lefta = cv2.imread('/tmp2/r07922076/CV_final/color/Real/TL0.png')
    # righta = cv2.imread('/tmp2/r07922076/CV_final/color/Real/TR0.png')
    img_rgb_out = gray2color('/tmp2/r07922076/CV_final/color/Real/TL0.png', lefta)
    plt.imsave('test.png', img_rgb_out)