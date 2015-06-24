import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t

cpdef get_patch(cntr_ptx, cntr_pty, np.ndarray img, patch_size = 9):
    
    cdef int x = cntr_ptx
    cdef int y = cntr_pty
    cdef int p = patch_size // 2
    return img[x-p:x+p+1, y-p:y+p+1]
    
cpdef copy_patch(np.ndarray[DTYPE_t, ndim=3] patch1, np.ndarray[DTYPE_t, ndim=3] patch2):
    
    unfilled = np.where(patch1[:,:,0] == 0.1111)
    cdef np.ndarray xx = unfilled[0]
    cdef np.ndarray yy = unfilled[1]
    cdef int i = 0

    while i <= len(xx) - 1:
        patch1[xx[i]][yy[i]] = patch2[xx[i]][yy[i]]
        i += 1
    return patch1

cpdef find_max_priority(np.ndarray boundary_ptx, np.ndarray boundary_pty, np.ndarray confidence_image, np.ndarray grad, np.ndarray norm, patch_size = 9):
    
    cdef float conf = np.sum(get_patch(boundary_ptx[0], boundary_pty[0], confidence_image))/patch_size
    cdef float max_grad = np.max(np.linalg.qr(get_patch(boundary_ptx[0], boundary_pty[0], grad))[0])
    cdef float norm_v = norm[boundary_ptx[0]][boundary_pty[0]]
    cdef float max = conf*max_grad*norm_v
    cdef int i = 0
    cdef float curr_data = 0, curr_conf = 0, curr_grad = 0
    
    while i < len(boundary_ptx):
        curr_patch = get_patch(boundary_ptx[i],boundary_pty[i], confidence_image)
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

cpdef patch_ncc(np.ndarray patch1, np.ndarray patch2):
    
    cdef int i = 0
    patch1[np.where(patch1 == 0.1111)] = 0.0
    cdef float ncc = np.sum((patch1/np.linalg.norm(patch1)) * (patch2/np.linalg.norm(patch2)))
    return ncc
    
cpdef patch_ssd(np.ndarray[DTYPE_t, ndim=3] patch1, np.ndarray[DTYPE_t, ndim=3] patch2):
    
    cdef int len = patch1.flatten().shape[0]
    cdef int i = 0
    cdef np.ndarray[DTYPE_t, ndim=1] patch_1 = patch1.flatten()
    cdef np.ndarray[DTYPE_t, ndim=1] patch_2 = patch2.flatten()
    cdef float sum = 0
    while i <= len - 1:
        if patch_1[i] != 0.1111:
            sum += (patch_1[i] - patch_2[i]) ** 2
        i += 1
    return sum

cpdef find_exemplar_patch_ncc(np.ndarray img, x, y, np.ndarray patch, patch_size = 9):
    
    cdef int offset = patch_size // 2
    cdef unsigned int x_boundary = img.shape[0]
    cdef unsigned int y_boundary = img.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=3] img_copy = img[offset:x_boundary, offset:y_boundary]
    cdef int i = 0
    filled_r = np.where(img_copy[:,:,0] != 0.1111)
    cdef np.ndarray xx = filled_r[0]
    cdef np.ndarray yy = filled_r[1]
    
    cdef np.ndarray[DTYPE_t, ndim=3] exemplar_patch = get_patch(5, 5, img)
    cdef float max_ncc = patch_ncc(exemplar_patch, patch)
    cdef np.ndarray[DTYPE_t, ndim=3] best_patch = exemplar_patch

    while i < len(xx) - 1:
        exemplar_patch = get_patch(xx[i], yy[i], img)
        if exemplar_patch.shape[0] == 9 and exemplar_patch.shape[1] == 9 and exemplar_patch.shape[2] == 3:
            if xx[i] != x and yy[i] != y and np.where(exemplar_patch == 0.1111)[0].shape[0] == 0:
                ncc = patch_ncc(exemplar_patch, patch)
                if ncc > max_ncc:
                    best_patch = exemplar_patch
                    print best_patch
                    x = xx[i]
                    y = yy[i]
                    max_ncc = ncc
                    print max_ncc,x,y
        i += 1
    return best_patch, x, y
    
cpdef find_exemplar_patch_ssd(np.ndarray[DTYPE_t, ndim=3] img, x, y, np.ndarray[DTYPE_t, ndim=3] patch, patch_size = 9):
    
    cdef int offset = patch_size // 2
    cdef unsigned int x_boundary = img.shape[0]
    cdef unsigned int y_boundary = img.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=3] img_copy = img[offset:x_boundary, offset:y_boundary]
    cdef int i = 0
    filled_r = np.where(img_copy[:,:,0] != 0.1111)
    cdef np.ndarray xx = filled_r[0]
    cdef np.ndarray yy = filled_r[1]
    
    cdef np.ndarray[DTYPE_t, ndim=3] exemplar_patch = get_patch(5, 5, img)
    cdef float min_ssd = patch_ssd(exemplar_patch, patch)
    cdef np.ndarray[DTYPE_t, ndim=3] best_patch = exemplar_patch

    while i < len(xx) - 1:
        exemplar_patch = get_patch(xx[i], yy[i], img)
        if xx[i] != x and yy[i] != y and np.where(exemplar_patch == 0.1111)[0].shape[0] == 0:
            ssd = patch_ssd(exemplar_patch, patch)
            if ssd < min_ssd and ssd != 0.0:
                best_patch = exemplar_patch
                print best_patch
                x = xx[i]
                y = yy[i]
                min_ssd = ssd
                print min_ssd,x,y
        i += 1
    return best_patch, x, y
    