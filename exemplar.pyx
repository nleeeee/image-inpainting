import numpy as np
cimport numpy as np
from libc.math cimport sqrt

ctypedef np.float64_t DTYPE_t
ctypedef np.int_t DTYPEi_t

cpdef get_patch(cntr_ptx, cntr_pty, np.ndarray img, patch_size = 9):
    
    cdef:
        int x = cntr_ptx
        int y = cntr_pty
        int p = patch_size // 2
        
    return img[x-p:x+p+1, y-p:y+p+1]
    
cpdef copy_patch(np.ndarray[DTYPE_t, ndim=3] patch1, 
                 np.ndarray[DTYPE_t, ndim=3] patch2):
    
    unfilled = np.where(patch1[:,:,1] == 1.0)
    
    cdef:
        np.ndarray[DTYPEi_t, ndim=1] xx = unfilled[0]
        np.ndarray[DTYPEi_t, ndim=1] yy = unfilled[1]
        int i = 0

    while i <= len(xx) - 1:
        patch1[xx[i]][yy[i]] = patch2[xx[i]][yy[i]]
        i += 1
        
    return patch1

cpdef find_max_priority(np.ndarray[DTYPEi_t, ndim=1] boundary_ptx, 
                        np.ndarray[DTYPEi_t, ndim=1] boundary_pty, 
                        np.ndarray[DTYPE_t, ndim=2] confidence_image, 
                        np.ndarray[DTYPE_t, ndim=2] dx,
                        np.ndarray[DTYPE_t, ndim=2] dy,
                        np.ndarray nx,
                        np.ndarray ny,
                        patch_size = 9,
                        alpha = 255.0):

    cdef: 
        float conf = np.sum(get_patch(boundary_ptx[0],
                                      boundary_pty[0], 
                                      confidence_image, 
                                      patch_size))/(patch_size ** 2)
        np.ndarray[DTYPE_t, ndim=2] grad = np.hypot(dx, dy)
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
        float max = conf * (data / alpha)
        int i = 0
        float curr_data = 0, curr_conf = 0, curr_grad = 0
    
    while i < len(boundary_ptx):
        curr_patch = get_patch(boundary_ptx[i],
                               boundary_pty[i], 
                               confidence_image, 
                               patch_size)
        curr_conf = np.sum(curr_patch)/(patch_size ** 2)
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
    
cpdef patch_ssd(np.ndarray[DTYPE_t, ndim=3] patch1, 
                np.ndarray[DTYPE_t, ndim=3] patch2):

    cdef:
        int m = patch1.shape[0]
        int n = patch1.shape[1]
        np.ndarray[DTYPE_t, ndim=3] patch3 = patch2[:m, :n, :]
        np.ndarray[DTYPE_t, ndim=1] patch1r = patch1[:,:,0].flatten()
        np.ndarray[DTYPE_t, ndim=1] patch1g = patch1[:,:,1].flatten()
        np.ndarray[DTYPE_t, ndim=1] patch1b = patch1[:,:,2].flatten()
        np.ndarray[DTYPE_t, ndim=1] patch2r = patch3[:,:,0].flatten()
        np.ndarray[DTYPE_t, ndim=1] patch2g = patch3[:,:,1].flatten()
        np.ndarray[DTYPE_t, ndim=1] patch2b = patch3[:,:,2].flatten()
        int i = 0
        int len = patch1r.shape[0]
        float sum = 0

    while i <= len - 1:
        if (patch1r[i] != 0.0 and
            patch1g[i] != 1.0 and
            patch1b[i] != 0.0): # ignore unfilled pixels 
            sum += (patch1r[i] - patch2r[i]) ** 2
            sum += (patch1g[i] - patch2g[i]) ** 2
            sum += (patch1b[i] - patch2b[i]) ** 2
        i += 1
    return sum
    
cpdef find_exemplar_patch_ssd(np.ndarray[DTYPE_t, ndim=3] img, 
                              x, 
                              y, 
                              np.ndarray[DTYPE_t, ndim=3] patch, 
                              patch_size = 9):
    
    cdef: 
        int offset = patch_size // 2
        unsigned int x_boundary = img.shape[0]
        unsigned int y_boundary = img.shape[1]
        np.ndarray[DTYPE_t, ndim=3] img_copy = img[offset:x_boundary-offset+1, 
                                                   offset:y_boundary-offset+1]
        unsigned int i = 0
    
    filled_r = np.where(img_copy[:,:,1] != 1.0)
    
    cdef: 
        np.ndarray[DTYPEi_t, ndim=1] xx = filled_r[0]
        np.ndarray[DTYPEi_t, ndim=1] yy = filled_r[1]
        float min_ssd = np.inf

    while i < len(xx) - 1:
        exemplar_patch = get_patch(xx[i] + offset, yy[i] + offset, img, patch_size)
        if (exemplar_patch.shape[0] == patch_size and 
            exemplar_patch.shape[1] == patch_size):
            if ((xx[i] + offset) != x and 
                (yy[i] + offset) != y and 
                np.where(exemplar_patch[:,:,1] == 1.0)[0].shape[0] == 0):
                ssd = patch_ssd(patch, exemplar_patch)
                if ssd < min_ssd:
                    best_patch = exemplar_patch
                    best_x = xx[i] + offset
                    best_y = yy[i] + offset
                    min_ssd = ssd
        i += 1
    return best_patch, best_x, best_y