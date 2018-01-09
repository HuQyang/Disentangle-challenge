"""
Some codes from https://github.com/Newmu/dcgan_code
"""
import math
import pprint
import scipy.misc
import numpy as np
import cv2

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def get_image(image_path, image_size):
    return transform(imread(image_path), image_size)

def save_images(images, grid_size, image_path, invert=True, channels=3,angle=None):
    if invert:
        images = inverse_transform(images)
    return imsave(images, grid_size, image_path, channels,angle)

def imread(path):
    return scipy.misc.imread(path).astype(np.float)

def imsave(images, grid_size, path, channels,angle=None):
    h, w = int(images.shape[1]), int(images.shape[2])
    img = np.zeros((h * int(grid_size[0]), w * int(grid_size[1]), channels))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx, image in enumerate(images):
        i = int(idx % grid_size[1])
        j = int(idx // grid_size[1])
        
        if angle is not None: 
            text = angle[idx] 
                 
            cv2.putText(image,np.array2string(text),(10,10), font, 0.5,(0,0,0),2,cv2.LINE_AA)
           
        if channels == 1:
            img[j*h:j*h+h, i*w:i*w+w, 0] = image 
        else:
            img[j*h:j*h+h, i*w:i*w+w, :] = image
    # Flatten third dimension if only grayscale image (necessary for scipy image save method).
    if channels == 1:
        img = img.reshape(img.shape[0:2])
    
    return scipy.misc.imsave(path, img)

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def transform(image, npx=64):
    # npx : # of pixels width/height of image
    cropped_image = center_crop(image, npx)
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.
