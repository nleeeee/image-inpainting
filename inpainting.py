from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
#from scipy.ndimage import filters
from PIL import Image
from scipy import ndimage
from skimage.morphology import erosion, disk
from sklearn.preprocessing import normalize
from exemplar import *

def paste_patch(x, y, patch, img, patch_size = 9):
    
    p = patch_size // 2
    img[x-p:x+p+1, y-p:y+p+1] = patch
    return img
    
def update(x, y, confidence_image, mask, patch_size = 9):
    
    p = patch_size // 2
    confidence_image[x-p:x+p+1, y-p:y+p+1] = 1
    mask[x-p:x+p+1, y-p:y+p+1] = 1
    return confidence_image, mask
    
if __name__ == '__main__':
    
    src = imread('input.jpg')
    mask = imread('input-mask.bmp')
    src2 = src/255.0
    mask /= 255.0
    src2[np.where(mask==0)] = 0.1111
    
    # initialize confidence
    confidence_image = np.zeros(mask.shape)
    confidence_image[np.where(mask != 0)] = 1
    
    while np.where(src2 == 0.1111)[0].shape[0] != 0:
        grayscale = src[:,:,0]*.229 + src[:,:,1]*.587 + src[:,:,2]*.114
        fill_front = mask - erosion(mask, disk(1))
        boundary_ptx = np.where(fill_front > 0)[0]
        boundary_pty = np.where(fill_front > 0)[1]
        
        # sobel operators
        #grayscale = ndimage.gaussian_filter(grayscale, 1)
        dx = ndimage.sobel(grayscale, 0)
        dy = ndimage.sobel(grayscale, 1)
        nx = ndimage.sobel(fill_front, 0)
        ny = ndimage.sobel(fill_front, 1)
        grad_norm = np.hypot(-dx, dy) # gradient normal
        norm = np.hypot(ny, nx)
        #norm = np.hypot(dy, dx)
        norm = normalize(norm, axis=0, norm='l1')
        dx[np.where(mask == 0)] = 0.1111
        dy[np.where(mask == 0)] = 0.1111
        grayscale[np.where(mask == 0)] = 0.1111
        
        highest_priority = find_max_priority(boundary_ptx, 
                                             boundary_pty, 
                                             confidence_image, 
                                             -dy,
                                             dx,
                                             nx,
                                             ny)
        best_patch = find_exemplar_patch_ssd(src2, 
                                            highest_priority[1], 
                                            highest_priority[2],           
                                            get_patch(highest_priority[1],
                                                      highest_priority[2],
                                                      src2))
        c = copy_patch(get_patch(highest_priority[1],
                                 highest_priority[2],
                                 src2), 
                       get_patch(best_patch[1],
                                 best_patch[2],
                                 src2))
        src2 = paste_patch(highest_priority[1],
                           highest_priority[2],
                           c,
                           src2)
        confidence_image, mask = update(highest_priority[1],
                                        highest_priority[2], 
                                        confidence_image,
                                        mask)
        imsave('inpainted.jpg', src2)
        plt.show(imshow(src2))    