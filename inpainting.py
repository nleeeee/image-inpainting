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
    
    return 1
    
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
    
def compute_confidence(boundary_ptx, boundary_pty, confidence_image, patch_size = 9):
    
    max = np.sum(get_patch(boundary_ptx[0], boundary_pty[0], confidence_image))/patch_size
    i = 0
    while i < len(boundary_ptx):
        curr_patch = get_patch(boundary_ptx[i],boundary_pty[i], confidence_image)
        curr_max = np.sum(curr_patch)/patch_size
        if curr_max > max:
            max = curr_max
            x = boundary_ptx[i]
            y = boundary_pty[i]
        i += 1
    return max, x, y
    
def get_patch(cntr_ptx, cntr_pty, img, patch_size = 9):
    
    x = cntr_ptx
    y = cntr_pty
    p = patch_size // 2
    return img[x-p:x+p+1, y-p:y+p+1]
    
def ssd(patch1, patch2):

    return np.sum((patch1.flatten() - patch2.flatten()) ** 2)

def inpainting(region, size):
    
    return 1

if __name__ == '__main__':
    
    src = imread('input.jpg')
    mask = imread('input-mask.bmp')
    fill_front = canny(mask, 3)
    
    # initialize confidence
    confidence_image = zeros(mask.shape)
    confidence_image[np.where(mask != 0)] = 1
    
    unfilled = mask
    grayscale = src[:,:,0]*.229 + src[:,:,1]*.587 + src[:,:,2]*.114
    grayscale = canny(grayscale, 3, 50, 10)
    
    boundary_ptx = np.where(fill_front > 0)[0]
    boundary_pty = np.where(fill_front > 0)[1]

    grayscale[np.where(mask == 0)] = 0.1111  
    grayscale[np.where(fill_front > 0)] = 0.1111      
    
    #compute_confidence(boundary_ptx, boundary_pty, confidence_image)
    