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
from skimage.morphology import erosion, disk

def find_max_priority(boundary_ptx, boundary_pty, confidence_image, grad, norm, patch_size = 9):
    
    conf = np.sum(get_patch(boundary_ptx[0], boundary_pty[0], confidence_image))/patch_size
    max_grad = np.max(np.linalg.qr(get_patch(boundary_ptx[0], boundary_pty[0], grad))[0])
    norm_v = norm[boundary_ptx[0]][boundary_pty[0]]
    max = conf*max_grad*norm_v
    i = 0
    while i < len(boundary_ptx):
        curr_patch = get_patch(boundary_ptx[i],boundary_pty[i], src)
        curr_conf = np.sum(curr_patch)/patch_size
        curr_grad = np.max(get_patch(boundary_ptx[i],boundary_pty[i],grad))
        norm_v = norm[boundary_ptx[i]][boundary_pty[i]]
        curr_data = curr_grad*norm_v
        curr_p = curr_conf*curr_data
        if curr_p > max:
            max = curr_p
            x = boundary_ptx[i]
            y = boundary_pty[i]
        i += 1
    return max, x, y
    
def find_max_priority2(boundary_ptx, boundary_pty, priority_image, patch_size = 9):
    
    max = np.sum(get_patch(boundary_ptx[0], boundary_pty[0], priority_image))/patch_size
    i = 1
    while i < len(boundary_ptx):
        curr_max = np.sum(get_patch(boundary_ptx[i],boundary_pty[i], priority_image))/patch_size
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
    
def patch_ncc(patch1, patch2):
    
    i = 0
    patch1[np.where(patch1==0.1111)] = 0
    ncc = np.sum((patch1/np.linalg.norm(patch1)) * (patch2/np.linalg.norm(patch2)))
    return ncc
    
def find_exemplar_patch(img, x, y, patch, patch_size = 9L):
    
    i = 0    
    filled = np.where(img != 0.1111)
    xx = filled[0]
    yy = filled[1]
    initialized = 0
    
    while i < len(xx) - 1:
        exemplar_patch = get_patch(xx[i], yy[i], img)
        print i
        if exemplar_patch.shape == (9, 9):
            if xx[i] != x and yy[i] != y and np.where(exemplar_patch==0.1111)[0].shape[0] == 0:
                if initialized == 0:
                    min_ssd = patch_ssd(get_patch(xx[i], yy[i], img), patch)
                    exemplar_patch = get_patch(xx[i], yy[i], img)
                    best_patch = exemplar_patch
                    initialized = 1
                ssd = patch_ssd(exemplar_patch, patch)
                if ssd > min_ssd and ssd != 0.0:
                    best_patch = exemplar_patch
                    x = xx[i]
                    y = yy[i]
                    min_ssd = ssd
                    print min_ssd,x,y
        i += 1
    return best_patch, x, y
    
def find_exemplar_patch2(img, x, y, patch):
    
    i = 0    
    filled_r = np.where(img[:,:,0] != 0.1111)
    xx = filled_r[0]
    yy = filled_r[1]
    initialized = 0
    
    while i < len(xx) - 1:
        exemplar_patch = get_patch(xx[i], yy[i], img)
        print i
        if exemplar_patch.shape == (9, 9, 3):
            if xx[i] != x and yy[i] != y and np.where(exemplar_patch==0.1111)[0].shape[0] == 0:
                if initialized == 0:
                    min_ssd = patch_ssd(get_patch(xx[i], yy[i], img), patch)
                    exemplar_patch = get_patch(xx[i], yy[i], img)
                    best_patch = exemplar_patch
                    initialized = 1
                ssd = patch_ssd(exemplar_patch, patch)
                if ssd < min_ssd and ssd != 0.0:
                    best_patch = exemplar_patch
                    x = xx[i]
                    y = yy[i]
                    min_ssd = ssd
                    print min_ssd,x,y
        i += 1
    return best_patch, x, y
    
def find_exemplar_patch3(img, x, y, patch):
    
    i = 0    
    filled_r = np.where(img[:,:,0] != 0.1111)
    xx = filled_r[0]
    yy = filled_r[1]
    initialized = 0
    
    while i < len(xx) - 1:
        exemplar_patch = get_patch(xx[i], yy[i], img)
        print i
        if exemplar_patch.shape == (9, 9, 3):
            if xx[i] != x and yy[i] != y and np.where(exemplar_patch==0.1111)[0].shape[0] == 0:
                if initialized == 0:
                    max_ncc = patch_ncc(get_patch(xx[i], yy[i], img), patch)
                    exemplar_patch = get_patch(xx[i], yy[i], img)
                    best_patch = exemplar_patch
                    initialized = 1
                ncc = patch_ncc(exemplar_patch, patch)
                if ncc > max_ncc and ncc != 0.0:
                    best_patch = exemplar_patch
                    x = xx[i]
                    y = yy[i]
                    min_ncc = ncc
                    print min_ncc,x,y
        i += 1
    return best_patch, x, y
    
def copy_patch(patch1, patch2):
    
    unfilled = np.where(patch1[:,:,0] == 0.1111)
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
    
    src = imread('golf.jpg')
    src2 = src/255.0
    mask = imread('golf-mask.pgm')
    mask /= 255.0
    src2[np.where(mask==0)] = 0.1111
    
    # initialize confidence
    confidence_image = np.zeros(mask.shape)
    confidence_image[np.where(mask != 0)] = 1
    
    # while np.where(src2 == 0.1111)[0].shape[0] != 0:
    #     grayscale = src[:,:,0]*.229 + src[:,:,1]*.587 + src[:,:,2]*.114
    #     fill_front = canny(mask, 1)
    #     boundary_ptx = np.where(fill_front > 0)[0]
    #     boundary_pty = np.where(fill_front > 0)[1]
    #     
    #     # sobel operators
    #     #grayscale = ndimage.gaussian_filter(grayscale, 1)
    #     dx = ndimage.sobel(grayscale, 0)
    #     dy = ndimage.sobel(grayscale, 1)
    #     grad_norm = np.hypot(-dy, dx) # gradient normal
    #     norm = canny(mask, 1) # normal
    #     grad_norm[np.where(mask == 0)] = 0.1111
    #     grad_norm[np.where(fill_front > 0)] = 0.1111
    #     grayscale[np.where(mask == 0)] = 0.1111
    #     
    #     data_term = grad_norm*norm
    #     priority_image = confidence_image*data_term
    #     #highest_priority = find_max_priority(boundary_ptx, boundary_pty, confidence_image, grad, norm)
    #     highest_priority = find_max_priority(boundary_ptx, boundary_pty, confidence_image, grad_norm, norm)
    #     best_patch = find_exemplar_patch2(src2,highest_priority[1],highest_priority[2], get_patch(highest_priority[1],highest_priority[2],src2))
    #     c = copy_patch(get_patch(highest_priority[1],highest_priority[2],src2),get_patch(best_patch[1],best_patch[2],src2)) #test
    #     paste_patch(highest_priority[1],highest_priority[2],c,src2) #test
    #     confidence_image, mask = update(highest_priority[1],highest_priority[2],confidence_image,mask) #test
    #     imsave('inpainted.jpg', src2)    
    
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
        grad_norm[np.where(mask == 0)] = 0.1111
        grayscale[np.where(mask == 0)] = 0.1111
        
        highest_priority = find_max_priority(boundary_ptx, boundary_pty, confidence_image, grad_norm, norm)
        best_patch = find_exemplar_patch3(src2,highest_priority[1],highest_priority[2], get_patch(highest_priority[1],highest_priority[2],src2))
        c = copy_patch(get_patch(highest_priority[1],highest_priority[2],src2),get_patch(best_patch[1],best_patch[2],src2)) #test
        paste_patch(highest_priority[1],highest_priority[2],c,src2) #test
        confidence_image, mask = update(highest_priority[1],highest_priority[2],confidence_image,mask) #test
        imsave('inpainted.jpg', src2)
        