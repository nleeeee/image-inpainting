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
from canny import *

def find_max_priority(boundary_ptx, boundary_pty, src, patch_size = 9):
    
    max = np.sum(get_patch(boundary_ptx[0], boundary_pty[0], confidence_image))/patch_size
    i = 0
    while i < len(boundary_ptx):
        curr_patch = get_patch(boundary_ptx[i],boundary_pty[i], src)
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
    
def patch_ssd(patch1, patch2):
    
    len = patch1.flatten().shape[0]
    i = 0
    patch1 = patch1.flatten()
    patch2 = patch2.flatten()
    sum = 0
    while i <= len - 1:
        if patch1[i] != 0.1111:
            sum += (patch1[i] - patch2[i]) ** 2
        i += 1
    return sum
    
def find_exemplar_patch(src, x, y, patch, patch_size = 9L):
    
    i = 0    
    filled = np.where(src != 0.1111)
    xx = filled[0]
    yy = filled[1]
    initialized = 0
    
    while i < len(xx) - 1:
        exemplar_patch = get_patch(xx[i], yy[i], src)
        print i
        if exemplar_patch.flatten().shape[0] == 81:
            if xx[i] != x and yy[i] != y and np.where(exemplar_patch==0.1111)[0].shape[0] == 0:
                if initialized == 0:
                    min_ssd = patch_ssd(get_patch(xx[i], yy[i], src), patch)
                    exemplar_patch = get_patch(xx[i], yy[i], src)
                    best_patch = exemplar_patch
                    initialized = 1
                ssd = patch_ssd(exemplar_patch, patch)
                if ssd <= min_ssd:
                    best_patch = exemplar_patch
                    x = filled[0][i]
                    y = filled[1][i]
                    min_ssd = ssd
                    print min_ssd,x,y
        i += 1
    return best_patch, x, y
    
def copy_patch(patch1, patch2):
    
    unfilled = np.where(patch1 == 0.1111)
    xx = unfilled[0]
    yy = unfilled[1]
    i = 0

    while i <= len(xx) - 1:
        patch1[xx[i]][yy[i]] = patch2[xx[i]][yy[i]]
        i += 1
        
    return patch1
    
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
    src2 = src/255.0
    mask = imread('input-mask.bmp')
    mask /= 255.0
    src2[np.where(mask==0)] = 0.1111
    
    # initialize confidence
    confidence_image = zeros(mask.shape)
    confidence_image[np.where(mask != 0)] = 1
    
    # todo: create loop here
    try:
        while np.where(src2 == 0.1111)[0].shape[0] != 0:
            grayscale = src[:,:,0]*.229 + src[:,:,1]*.587 + src[:,:,2]*.114
            fill_front = canny(mask, 1)
            boundary_ptx = np.where(fill_front > 0)[0]
            boundary_pty = np.where(fill_front > 0)[1]
            
            # sobel operators
            dx = ndimage.sobel(grayscale, 0)
            dy = ndimage.sobel(grayscale, 1)
            grad = np.hypot(dx, dy) # gradient
            norm = np.hypot(-dy, dx) # normal
            grad[np.where(mask == 0)] = 0
            grad[np.where(fill_front > 0)] = 0
            
            grayscale /= 255.0
            grayscale[np.where(mask == 0)] = 0.1111
            xx = np.where(grayscale != 0.1111)[0]
            yy = np.where(grayscale != 0.1111)[1]
            
            data_term = np.linalg.qr(grad)[0]*norm
            priority_image = confidence_image*data_term
            highest_priority = find_max_priority(boundary_ptx, boundary_pty, priority_image)
            best_patch = find_exemplar_patch(grayscale,highest_priority[1],highest_priority[2], get_patch(highest_priority[1],highest_priority[2],grayscale))
            c = copy_patch(get_patch(highest_priority[1],highest_priority[2],src2),get_patch(best_patch[1],best_patch[2],src2)) #test 
            paste_patch(highest_priority[1],highest_priority[2],c,src2) #test
            confidence_image, mask = update(highest_priority[1],highest_priority[2],confidence_image,mask) #test
            imsave('inpainted.jpg', src2)
    except:
        imsave('inpainted.jpg', src2)
        
    