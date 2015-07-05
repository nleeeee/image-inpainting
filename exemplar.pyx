import numpy as np
cimport numpy as np
from libc.math cimport sqrt

ctypedef np.float64_t DTYPE_t
ctypedef np.int_t DTYPEi_t

cpdef get_patch(cntr_ptx, cntr_pty, np.ndarray img, patch_size):
    '''Gets the patch centered at x and y in the image img with dimensions
    patch_size by patch_size.
    
    Parameters
    ----------
    cntr_ptx : int
        x coordinate of the centre of the patch.
    cntr_pty : int 
        y coordinate of the centre of the patch.
    img: 2-D or 3-D array
        The image where patch is to be obtained from.
    patch_size : int
        Dimensions of the patch; must be odd.
        
    Returns
    -------
    patch : 2-D or 3-D array
        The patch of size patch_size by patch_size centered at (x,y) in img.
    '''
    
    cdef:
        int x = cntr_ptx
        int y = cntr_pty
        int p = patch_size // 2
        np.ndarray patch = img[x-p:x+p+1, y-p:y+p+1]
    return patch
    
cpdef copy_patch(np.ndarray[DTYPE_t, ndim=3] patch_dst, 
                 np.ndarray[DTYPE_t, ndim=3] patch_src):
    '''Copies the values from patch_src to patch_dst at where patch_dst has
    values specifying an unfilled region (unfilled regions have value of 
    [0.0, 1.0, 0.0]).
    
    Parameters
    ----------
    patch_dst : 3-D array
        The target patch with an unfilled region.
    patch_src : 3-D array
        The source patch with information to be copied to patch_dst.
        
    Returns
    -------
    patch_dst : 3-D array
        The target patch after it has been filled with information 
        from patch_src.
    '''
    
    # find locations of unfilled pixels
    unfilled_pixels = np.where(patch_dst[:,:,1] == 0.9999)
    
    cdef:
        # x coordinates of unfilled pixels
        np.ndarray[DTYPEi_t, ndim=1] unf_x = unfilled_pixels[0]
        # y coordinates of unfilled pixels
        np.ndarray[DTYPEi_t, ndim=1] unf_y = unfilled_pixels[1]
        int i = 0

    while i <= len(unf_x) - 1:
        patch_dst[unf_x[i]][unf_y[i]] = patch_src[unf_x[i]][unf_y[i]]
        i += 1
    
    return patch_dst

cpdef find_max_priority(np.ndarray[DTYPEi_t, ndim=1] boundary_ptx, 
                        np.ndarray[DTYPEi_t, ndim=1] boundary_pty, 
                        np.ndarray[DTYPE_t, ndim=2] confidence, 
                        np.ndarray[DTYPE_t, ndim=2] dx,
                        np.ndarray[DTYPE_t, ndim=2] dy,
                        np.ndarray nx,
                        np.ndarray ny,
                        patch_size,
                        alpha = 255.0):
    '''Finds the patch centered at pixels along the fill front which has the 
    highest priority value.
    
    Parameters
    ----------
    boundary_ptx : 1-D array
        An array of x coordinates specifying the locations of the pixels of
        the boundary of the fill region.
    boundary_pty : 1-D array 
        An array of y coordinates specifying the locations of the pixels of
        the boundary of the fill region.
    confidence : 2-D array
        2-D array holding confidence values for the image to be inpainted.
    dx : 2-D array
        The gradient image of the unfilled image in x direction.
    dy : 2-D array
        The gradient image of the unfilled image in y direction.
    nx : 2-D array
        The normal image of the mask in x direction.
    ny : 2-D array
        The normal image of the mask in y direction.
    patch_size : int
        Dimensions of the patch; must be odd.
    alpha : 
        The normalization factor; suggested value of 255 as described by
        Criminisi.
    
        
    Returns
    -------
    max : float
        The highest priority value.
    x : int
        x coordinate of the center of the patch at which the highest priority
        value was computed
    y : int
        y coordinate of the center of the patch at which the highest priority
        value was computed
    '''
    # initialize first priority value
    cdef: 
        float conf = np.sum(get_patch(boundary_ptx[0],
                                      boundary_pty[0], 
                                      confidence, 
                                      patch_size))/(patch_size ** 2) # confidence value
        np.ndarray[DTYPE_t, ndim=2] grad = np.hypot(dx, dy)
        # a gradient has value of 0 on the boundary;
        # so get the maximum gradient magnitude in a patch
        np.ndarray[DTYPE_t, ndim=2] grad_patch = abs(get_patch(boundary_ptx[0],
                                                               boundary_pty[0],
                                                               grad,
                                                               patch_size))
    cdef:
        int xx = np.where(grad_patch == np.max(grad_patch))[0][0]
        int yy = np.where(grad_patch == np.max(grad_patch))[1][0]
        float max_gradx = dx[xx][yy]
        float max_grady = dy[xx][yy]
        float Nx = nx[boundary_ptx[0]][boundary_pty[0]]
        float Ny = ny[boundary_ptx[0]][boundary_pty[0]]
    
        int x = boundary_ptx[0]
        int y = boundary_pty[0]
        
        float data = abs(max_gradx * Nx + max_grady * Ny)
        
    if (Nx ** 2 + Ny ** 2) != 0:
        data /= (Nx ** 2 + Ny ** 2)
        
    cdef:
        float max = conf * (data / alpha) # initial priority value
        int i = 0
        float curr_data = 0, curr_conf = 0, curr_grad = 0
    # iterate through all patches centered at a pixel on the boundary of 
    # unfilled region to find the patch with the highest priority value
    while i < len(boundary_ptx):
        curr_patch = get_patch(boundary_ptx[i],
                               boundary_pty[i], 
                               confidence, 
                               patch_size)
        curr_conf = np.sum(curr_patch)/(patch_size ** 2) # confidence value
        # a gradient has value of 0 on the boundary;
        # so get the maximum gradient magnitude in a patch
        grad_patch = abs(get_patch(boundary_ptx[i],
                                   boundary_pty[i],
                                   grad,
                                   patch_size))
        xx = np.where(grad_patch == np.max(grad_patch))[0][0]
        yy = np.where(grad_patch == np.max(grad_patch))[1][0]
        max_gradx = dx[xx][yy]
        max_grady = dy[xx][yy]            
        Nx = nx[boundary_ptx[i]][boundary_pty[i]]
        Ny = ny[boundary_ptx[i]][boundary_pty[i]]
        curr_data = abs(max_gradx * Nx + max_grady * Ny)
        if (Nx ** 2 + Ny ** 2) != 0:
            curr_data /= (sqrt(Nx ** 2 + Ny ** 2))
        curr_p = curr_conf * (curr_data / alpha)
        if curr_p > max:
            max = curr_p
            x = boundary_ptx[i]
            y = boundary_pty[i]
        i += 1
    return max, x, y
    
cpdef patch_ssd(np.ndarray[DTYPE_t, ndim=3] patch_dst, 
                np.ndarray[DTYPE_t, ndim=3] patch_src):
    '''Computes the sum of squared differences between patch_dst and patch_src
    at every pixel.
    
    Parameters
    ----------
    patch_dst : 3-D array
        The patch with an unfilled region.
    patch_src : 3-D array
        The patch being compared to patch_dst. 
        
    Returns
    -------
    sum : float
        The sum of squared differences value of patch_dst and patch_src.
    '''
    
    cdef:
        int m = patch_dst.shape[0]
        int n = patch_dst.shape[1]
        # ensure two patches are of same dimensions
        np.ndarray[DTYPE_t, ndim=3] patch_srcc = patch_src[:m, :n, :]
        np.ndarray[DTYPE_t, ndim=1] patch_dst_r = patch_dst[:,:,0].flatten()
        np.ndarray[DTYPE_t, ndim=1] patch_dst_g = patch_dst[:,:,1].flatten()
        np.ndarray[DTYPE_t, ndim=1] patch_dst_b = patch_dst[:,:,2].flatten()
        np.ndarray[DTYPE_t, ndim=1] patch_src_r = patch_srcc[:,:,0].flatten()
        np.ndarray[DTYPE_t, ndim=1] patch_src_g = patch_srcc[:,:,1].flatten()
        np.ndarray[DTYPE_t, ndim=1] patch_src_b = patch_srcc[:,:,2].flatten()
        int i = 0
        int len = patch_dst_r.shape[0]
        float sum = 0

    while i <= len - 1:
        if (patch_dst_r[i] != 0.0 and
            patch_dst_r[i] != 0.9999 and
            patch_dst_r[i] != 0.0): # ignore unfilled pixels 
            sum += (patch_dst_r[i] - patch_src_r[i]) ** 2
            sum += (patch_dst_g[i] - patch_src_g[i]) ** 2
            sum += (patch_dst_b[i] - patch_src_b[i]) ** 2
        i += 1
    return sum
    
cpdef find_exemplar_patch_ssd(np.ndarray[DTYPE_t, ndim=3] img, 
                              x, 
                              y, 
                              np.ndarray[DTYPE_t, ndim=3] patch, 
                              patch_size = 9):
    '''Finds the best exemplar patch with the minimum sum of squared 
    differences.
    
    Parameters
    ----------
    img : 3-D array
        The image with unfilled regions to be inpainted.
    x : int
        The x coordinate of the center of the patch with tbe highest
        priority value.
    y : int
        The y coordinate of the center of the patch with tbe highest
        priority value.
    patch : 3-D array
        The patch centered at (x, y) with the highest priority value and an 
        unfilled region.
    patch_size : int
        Dimensions of the patch size; must be odd.
        
    Returns
    -------
    best_patch : 3-D array
        The argmin patch with the lowest ssd with patch.
    best_x : int
        The x coordinate which best_patch is centered at.
    best_y : int
        The y coordinate which best_patch is centered at.
    '''
    
    cdef: 
        int offset = patch_size // 2
        int x_boundary = img.shape[0]
        int y_boundary = img.shape[1]
        int i = 0
        float min_ssd = np.inf
        # offset the borders of the image by offset pixels to avoid
        # looking through patches that are out of the region
        np.ndarray[DTYPE_t, ndim=3] img_copy = img[offset:x_boundary-offset+1, 
                                                   offset:y_boundary-offset+1]
        
    # locations of the unfilled region
    filled_r = np.where(img_copy[:,:,1] != 0.9999)
    
    cdef:
        # x coordinates of the unfilled region
        np.ndarray[DTYPEi_t, ndim=1] xx = filled_r[0]
        # y coordinates of the unfilled region
        np.ndarray[DTYPEi_t, ndim=1] yy = filled_r[1]
        
    while i < len(xx) - 1:
        exemplar_patch = get_patch(xx[i] + offset, yy[i] + offset, img, patch_size)
        if (exemplar_patch.shape[0] == patch_size and 
            exemplar_patch.shape[1] == patch_size):
            # check if we're not getting the same patch from the parameters
            # and if the potential exemplar patch has no unfilled regions
            if ((xx[i] + offset) != x and 
                (yy[i] + offset) != y and 
                np.where(exemplar_patch[:,:,1] == 0.9999)[0].shape[0] == 0):
                ssd = patch_ssd(patch, exemplar_patch)
                if ssd < min_ssd:
                    best_patch = exemplar_patch
                    best_x = xx[i] + offset
                    best_y = yy[i] + offset
                    min_ssd = ssd
        i += 1
    return best_patch, best_x, best_y