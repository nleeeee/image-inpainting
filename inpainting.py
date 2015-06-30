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
    '''Updates the confidence values and mask image for the image to be
    to be inpainted.
    
    Parameters
    ----------
    x : int
        x coordinate of the centre of the patch which has been filled.
    y : int 
        y coordinate of the centre of the patch which has been filled.
    patch : 3-D array
        The patch which has been filled with information from the exemplar patch
    img : 3-D array
        The target image with the unfilled regions.
    patch_size : int
        Dimensions of the patch; must be odd.
        
    Returns
    -------
    img : 3-D array
        The target image after it has been updated with the new filled patch.
    '''
    
    p = patch_size // 2
    img[x-p:x+p+1, y-p:y+p+1] = patch
    return img
    
def update(x, y, confidence, mask, patch_size = 9):
    '''Updates the confidence values and mask image for the image to be
    to be inpainted.
    
    Parameters
    ----------
    x : int
        x coordinate of the centre of the patch which has been filled.
    y : int 
        y coordinate of the centre of the patch which has been filled.
    confidence : 2-D array
        2-D array holding confidence values for the image to be inpainted.
    mask : 2-D array
        A binary image specifying regions to be inpainted with a value of 0 
        and 1 elsewhere.
    patch_size : int
        Dimensions of the patch; must be odd.
        
    Returns
    -------
    confidence : 2-D array
        2-D array holding confidence values for the image to be inpainted.
    mask : 2-D array
        A binary image specifying regions to be inpainted with a value of 0 
        and 1 elsewhere.
    '''
    
    p = patch_size // 2
    confidence[x-p:x+p+1, y-p:y+p+1] = 1
    mask[x-p:x+p+1, y-p:y+p+1] = 1
    return confidence, mask
    
if __name__ == '__main__':
    
    src = imread('golf.jpg') # source image
    mask = imread('golf-mask.pgm') # mask; binary image specifying unfilled regions with a value of 0
    mask /= 255.0
    unfilled_img = src/255.0
    
    # initialize confidence
    confidence = np.zeros(mask.shape)
    confidence[np.where(mask != 0)] = 1
    
    unfilled_img[np.where(mask == 0.0)] = [0.0, 1.0, 0.0] # place holder value for unfilled pixels
    
    patch_count = 0
    patch_size = 11 # must be odd
    
    grayscale = src[:,:,0]*.2125 + src[:,:,1]*.7154 + src[:,:,2]*.0721
    grayscale /= 255.0
    grayscale = ndimage.gaussian_filter(grayscale, 0.5) # gaussian smoothing for computing gradients
    
    while mask.any():
        fill_front = mask - erosion(mask, disk(1)) # boundary of unfilled region
        # pixels where the fill front is located
        boundary_ptx = np.where(fill_front > 0)[0] # x coordinates
        boundary_pty = np.where(fill_front > 0)[1] # y coordinates
        
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
                                             confidence, 
                                             -dy,
                                             dx,
                                             nx,
                                             ny,
                                             patch_size)
                                             
        max_x = highest_priority[1]
        max_y = highest_priority[2]
        max_patch = get_patch(max_x, max_y, unfilled_img, patch_size)
        
        best_patch = find_exemplar_patch_ssd(unfilled_img, 
                                             max_x, 
                                             max_y,           
                                             max_patch,
                                             patch_size)
        copied_patch = copy_patch(max_patch, best_patch[0])
        unfilled_img = paste_patch(max_x, 
                                   max_y, 
                                   copied_patch, 
                                   unfilled_img, 
                                   patch_size)
        confidence_image, mask = update(max_x,
                                        max_y, 
                                        confidence,
                                        mask,
                                        patch_size)
        patch_count += 1
        print patch_count, 'patches inpainted', highest_priority[1:], '<-', best_patch[1:]
        imsave('inpainted.jpg', unfilled_img)
        
    plt.show(imshow(src2))    