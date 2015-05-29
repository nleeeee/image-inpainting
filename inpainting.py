from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from canny import canny

def compute_normal(boundary_pt, normal, patch_size = 9):
    
    x = boundary_pt[0]
    y = boundary_pt[1]
    pp = np.arange(-1*patch_size, patch_size+1)
    norm = 0
    
    for p in pp:
        for q in pp:
            norm += normal[x+p,y+q]
    return norm
    
    
def compute_data(boundary_pt, alpha, gradient, front_normal, patch_size = 9):
    
    x = boundary_pt[0]
    y = boundary_pt[1]
    pp = np.arange(-1*patch_size, patch_size+1)
    data = 0
    
    for p in pp:
        for q in pp:
            data += np.dot(gradient[x+p,y+q], front_normal[x+p,y+q])
    data /= alpha
    return data
    
    
def compute_confidence(boundary_pt, confidence_image, unfilled, patch_size = 9):
    
    #return np.sum(get_patch(boundary_pt, confidence_image))/patch_size
    '''
    x = boundary_pt[0]
    y = boundary_pt[1]

    pp = np.arange(-1*patch_size, patch_size+1)
    confidence = 0
    
    for p in pp:
        for q in pp:
            confidence += unfilled[x+p,y+q]
    confidence /= patch_size
    confidence_image[x,y] = confidence
    return confidence_image
    '''
    return 1

    
def get_patch(cntr_pt, img, patch_size = 9):
    
    x = cntr_pt[0]
    y = cntr_pt[1]
    p = patch_size // 2
    return img[x-p:x+p+1, y-p:y+p+1]


def inpainting(region, size):
    
    return 1


if __name__ == '__main__':
    
    src = imread('input.jpg')
    mask = imread('input-mask.bmp')
    fill_front = canny(mask, 3)
    
    # initializa confidence
    confidence_image = zeros(mask.shape)
    confidence_image[np.where(mask!=0)] = 1
    
    unfilled = mask
    grad = src[:,:,0]*.229 + src[:,:,1]*.587 + src[:,:,2]*.114
    region = src
    region[np.where(mask == 0)] = 255
    grayscale = region[:,:,0]*.229 + region[:,:,1]*.587 + region[:,:,2]*.114
    grayscale = canny(grayscale, 3, 50, 10)
    
