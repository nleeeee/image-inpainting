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
    
    unfilled = np.where(patch1[:,:,0] == 0.00001111)
    
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
    
    dx[np.where(dx == 0.00001111)] = 0.0
    dy[np.where(dy == 0.00001111)] = 0.0
    
    cdef: 
        float conf = np.sum(get_patch(boundary_ptx[0],
                                      boundary_pty[0], 
                                      confidence_image, 
                                      patch_size))/(patch_size ** 2)
        np.ndarray[DTYPE_t, ndim=2] gradx = abs(get_patch(boundary_ptx[0], 
                                                          boundary_pty[0], 
                                                          dx, 
                                                          patch_size))
        np.ndarray[DTYPE_t, ndim=2] grady = abs(get_patch(boundary_ptx[0], 
                                                          boundary_pty[0], 
                                                          dy, 
                                                          patch_size))
        float Nx = nx[boundary_ptx[0]][boundary_pty[0]]
        float Ny = ny[boundary_ptx[0]][boundary_pty[0]]
    
        int x = boundary_ptx[0]
        int y = boundary_pty[0]
    '''
    np.ndarray Nx = get_patch(boundary_ptx[0], 
                                boundary_pty[0], 
                                nx, 
                                patch_size)
    np.ndarray Ny = get_patch(boundary_ptx[0], 
                                boundary_pty[0], 
                                ny, 
                                patch_size)
    '''
        
        
    cdef:
        float max_gradx = np.max(gradx)
        float max_grady = np.max(grady)
        float data = abs(max_gradx * Nx + max_grady * Ny) / alpha
        float max = conf * data
        int i = 0
        float curr_data = 0, curr_conf = 0, curr_grad = 0
    
    while i < len(boundary_ptx):
        curr_patch = get_patch(boundary_ptx[i],
                               boundary_pty[i], 
                               confidence_image, 
                               patch_size)
        curr_conf = np.sum(curr_patch)/(patch_size ** 2)
        max_gradx = np.max(abs(get_patch(boundary_ptx[i],
                                         boundary_pty[i],
                                         dx, 
                                         patch_size)))
        max_grady = np.max(abs(get_patch(boundary_ptx[i],
                                         boundary_pty[i],
                                         dy, 
                                         patch_size)))
        Nx = nx[boundary_ptx[i]][boundary_pty[i]]
        Ny = ny[boundary_ptx[i]][boundary_pty[i]]
        curr_data = abs(max_gradx * Nx + max_grady * Ny) / alpha
        '''
        gradx = get_patch(boundary_ptx[i], 
                          boundary_pty[i], 
                          dx, 
                          patch_size)
        grady = get_patch(boundary_ptx[i], 
                          boundary_pty[i], 
                          dy, 
                          patch_size)
        Nx = get_patch(boundary_ptx[i], 
                       boundary_pty[i], 
                       nx, 
                       patch_size)
        Ny = get_patch(boundary_ptx[i], 
                       boundary_pty[i], 
                       ny, 
                       patch_size)
        
        curr_data = np.sum((max_gradx * Nx + max_grady * Ny))
        '''
        curr_p = curr_conf * curr_data
        if curr_p > max:
            max = curr_p
            x = boundary_ptx[i]
            y = boundary_pty[i]
        i += 1
    return max, x, y

cpdef patch_ncc(np.ndarray[DTYPE_t, ndim=3] patch1, 
                np.ndarray[DTYPE_t, ndim=3] patch2):
    
    cdef int i = 0
    patch1[np.where(patch1 == 0.00001111)] = 0.0
    cdef float ncc = np.sum((patch1/np.linalg.norm(patch1)) * (patch2/np.linalg.norm(patch2)))
    return ncc
    
cpdef patch_ssd(np.ndarray[DTYPE_t, ndim=3] patch1, 
                np.ndarray[DTYPE_t, ndim=3] patch2):
    
    cdef:
        np.ndarray[DTYPE_t, ndim=1] patch1r = patch1[:,:,0].flatten()
        np.ndarray[DTYPE_t, ndim=1] patch1g = patch1[:,:,1].flatten()
        np.ndarray[DTYPE_t, ndim=1] patch1b = patch1[:,:,2].flatten()
        np.ndarray[DTYPE_t, ndim=1] patch2r = patch2[:,:,0].flatten()
        np.ndarray[DTYPE_t, ndim=1] patch2g = patch2[:,:,1].flatten()
        np.ndarray[DTYPE_t, ndim=1] patch2b = patch2[:,:,2].flatten()
        int i = 0
        int len = patch1r.shape[0]
        float sum = 0
        
    while i <= len - 1:
        if patch1r[i] != 0.00001111: # ignore unfilled pixels 
            sum += (patch1r[i] - patch2r[i]) ** 2
            sum += (patch1g[i] - patch2g[i]) ** 2
            sum += (patch1b[i] - patch2b[i]) ** 2
        i += 1
    return sum

cpdef find_exemplar_patch_ncc(np.ndarray[DTYPE_t, ndim=3] img, 
                              x, 
                              y, 
                              np.ndarray[DTYPE_t, ndim=3] patch, 
                              patch_size = 9):
    
    cdef:
        int offset = patch_size // 2 
        unsigned int x_boundary = img.shape[0]
        unsigned int y_boundary = img.shape[1]
        np.ndarray[DTYPE_t, ndim=3] img_copy = img[offset:x_boundary-offset, 
                                                   offset:y_boundary-offset]
        unsigned int i = 0
    
    filled_r = np.where(img_copy[:,:,0] != 0.00001111)
    
    cdef:
        np.ndarray[DTYPEi_t, ndim=1] xx = filled_r[0]
        np.ndarray[DTYPEi_t, ndim=1] yy = filled_r[1]
    
        np.ndarray[DTYPE_t, ndim=3] exemplar_patch = get_patch(offset + 1, 
                                                               offset + 1, 
                                                               img, 
                                                               patch_size)
        float max_ncc = patch_ncc(exemplar_patch, patch)
        np.ndarray[DTYPE_t, ndim=3] best_patch = exemplar_patch

    while i < len(xx) - 1:
        exemplar_patch = get_patch(xx[i], yy[i], img, patch_size)
        if exemplar_patch.shape[0] == patch_size and exemplar_patch.shape[1] == patch_size and  exemplar_patch.shape[2] == 3:
            if xx[i] != x and yy[i] != y and np.where(exemplar_patch == 0.00001111)[0].shape[0] == 0:
                ncc = patch_ncc(exemplar_patch, patch)
                if ncc > max_ncc:
                    best_patch = exemplar_patch
                    x = xx[i]
                    y = yy[i]
                    max_ncc = ncc
        i += 1
    return best_patch, x, y
    
cpdef find_exemplar_patch_ssd(np.ndarray[DTYPE_t, ndim=3] img, 
                              x, 
                              y, 
                              np.ndarray[DTYPE_t, ndim=3] patch, 
                              patch_size = 9):
    
    cdef: 
        int offset = patch_size // 2
        unsigned int x_boundary = img.shape[0]
        unsigned int y_boundary = img.shape[1]
        np.ndarray[DTYPE_t, ndim=3] img_copy = img[offset:x_boundary-offset, 
                                                   offset:y_boundary-offset]
        unsigned int i = 0
    
    filled_r = np.where(img_copy[:,:,0] != 0.00001111)
    
    cdef: 
        np.ndarray[DTYPEi_t, ndim=1] xx = filled_r[0]
        np.ndarray[DTYPEi_t, ndim=1] yy = filled_r[1]
    
        np.ndarray[DTYPE_t, ndim=3] exemplar_patch = get_patch(offset + 1, 
                                                               offset + 1, 
                                                               img, 
                                                               patch_size)
        float min_ssd = patch_ssd(exemplar_patch, patch)
        np.ndarray[DTYPE_t, ndim=3] best_patch = exemplar_patch

    while i < len(xx) - 1:
        exemplar_patch = get_patch(xx[i], yy[i], img, patch_size)
        if exemplar_patch.shape[0] == patch_size and exemplar_patch.shape[1] == patch_size and  exemplar_patch.shape[2] == 3:
            if xx[i] != x and yy[i] != y and np.where(exemplar_patch == 0.00001111)[0].shape[0] == 0:
                ssd = patch_ssd(exemplar_patch, patch)
                if ssd < min_ssd and ssd != 0.0:
                    best_patch = exemplar_patch
                    x = xx[i]
                    y = yy[i]
                    min_ssd = ssd
        i += 1
    return best_patch, x, y