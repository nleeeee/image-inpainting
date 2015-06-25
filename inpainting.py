from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.ndimage import filters
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
    mask /= 255.0
    src2 = src/255.0
    
    # initialize confidence
    confidence_image = np.zeros(mask.shape)
    confidence_image[np.where(mask != 0)] = 1
    
    src2[np.where(mask == 0.0)] = 0.00001111 # place holder value for unfilled pixels
    
    patch_count = 0
    patch_size = 15 # must be odd
    
    while np.where(src2 == 0.00001111)[0].shape[0] != 0:
        #grayscale = src2[:,:,0]*.229 + src2[:,:,1]*.587 + src2[:,:,2]*.114
        fill_front = mask - erosion(mask, disk(1))
        # pixels where the fill front is located
        boundary_ptx = np.where(fill_front > 0)[0]
        boundary_pty = np.where(fill_front > 0)[1]
        
        #grayscale = ndimage.gaussian_filter(grayscale, 1) # gaussian smoothing prior to calculating gradients
        #grayscale[np.where(mask == 0.0)] = 0.0
        #plt.show(imshow(grayscale))
        
        # compute gradients with sobel operators
        
        dx_r = ndimage.sobel(src2[:,:,0], 0)
        dy_r = ndimage.sobel(src2[:,:,0], 1)
        dx_g = ndimage.sobel(src2[:,:,1], 0)
        dy_g = ndimage.sobel(src2[:,:,1], 1)
        dx_b = ndimage.sobel(src2[:,:,2], 0)
        dy_b = ndimage.sobel(src2[:,:,2], 1)
        dx = (dx_r + dx_g + dx_b) / 3
        dy = (dy_r + dy_g + dy_b) / 3
        

        #dx = ndimage.sobel(grayscale, 0)
        #dy = ndimage.sobel(grayscale, 1)
        
        # compute normals
        nx = ndimage.sobel(fill_front, 0)
        ny = ndimage.sobel(fill_front, 1)
        # normalize rows
        #nx /= nx.sum(axis=1)[:,np.newaxis]
        #ny /= ny.sum(axis=1)[:,np.newaxis]
        #nx = normalize(nx, norm='l1', axis=1)
        #ny = normalize(ny, norm='l1', axis=1)
        N = np.hypot(ny, nx)
        N = normalize(N, norm='l2', axis=0)
        nx /= N
        ny /= N
        dx[np.where(mask == 0)] = 0.00001111
        dy[np.where(mask == 0)] = 0.00001111
        
        highest_priority = find_max_priority(boundary_ptx, 
                                             boundary_pty, 
                                             confidence_image, 
                                             -dx,
                                             dy,
                                             ny,
                                             nx,
                                             patch_size)
        best_patch = find_exemplar_patch_ssd(src2, 
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