from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.ndimage import filters
from PIL import Image
from scipy import ndimage
from skimage.morphology import erosion, disk
from exemplar import *
from sklearn.preprocessing import normalize

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
    mask /= 255.0
    src2 = src/255.0
    
    # initialize confidence
    confidence_image = np.zeros(mask.shape)
    confidence_image[np.where(mask != 0)] = 1
    
    src2[np.where(mask == 0.0)] = [0.0, 1.0, 0.0] # place holder value for unfilled pixels
    
    patch_count = 0
    patch_size = 9 # must be odd
    
    grayscale = src[:,:,0]*.2125 + src[:,:,1]*.7154 + src[:,:,2]*.0721
    grayscale /= 255.0
    
    while mask.any():
        fill_front = mask - erosion(mask, disk(1))
        # pixels where the fill front is located
        boundary_ptx = np.where(fill_front > 0)[0]
        boundary_pty = np.where(fill_front > 0)[1]
        
        #grayscale = ndimage.gaussian_filter(grayscale, 0.5) # gaussian smoothing for computing gradients
        
        # compute gradients with sobel operators        
        dx = ndimage.sobel(grayscale, 0)
        dy = ndimage.sobel(grayscale, 1)
        # compute normals
        nx = ndimage.sobel(mask, 0)
        ny = ndimage.sobel(mask, 1)
        
        dx[np.where(mask == 0)] = 0.0
        dy[np.where(mask == 0)] = 0.0
        
        highest_priority = find_max_priority(boundary_ptx, 
                                             boundary_pty, 
                                             confidence_image, 
                                             -dy,
                                             dx,
                                             nx,
                                             ny,
                                             patch_size)
        best_patch = find_exemplar_patch_ncc(src2, 
                                             highest_priority[1], 
                                             highest_priority[2],           
                                             get_patch(highest_priority[1],
                                                       highest_priority[2],
                                                       src2,
                                                       patch_size),
                                             patch_size)
        c = copy_patch(get_patch(highest_priority[1],
                                 highest_priority[2],
                                 src2, 
                                 patch_size), 
                       best_patch[0])
        src2 = paste_patch(highest_priority[1],
                           highest_priority[2],
                           c,
                           src2,
                           patch_size)
        confidence_image, mask = update(highest_priority[1],
                                        highest_priority[2], 
                                        confidence_image,
                                        mask,
                                        patch_size)
        patch_count += 1
        print patch_count, 'patches inpainted', highest_priority[1:], '<-', best_patch[1:]
        imsave('inpainted.jpg', src2)
        
    plt.show(imshow(src2))    