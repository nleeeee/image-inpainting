# Image Inpainting

Implementaion of exemplar-based image inpainting algorithm by Criminisi et al. 

Requirements:

Python 2.7.9 or greater

Cython 0.22 or greater

NumPy for Python 2

SciPy for Python 2

Matplotlib for Python 2

#### Instructions

![1](instruction-pics/1.png)

Run the program to open the GUI.

![2](instruction-pics/2.png)

Enter the patch size. By default, it is 9. The patch size must be odd.

![3](instruction-pics/3.png)

Choose to whether to apply Gaussian filtering prior to computing the image gradients and choose sigma value.

![4](instruction-pics/4.png)

Load the image and its mask. Press the Inpaint button to run the algorithm.

![5](instruction-pics/5.png)

#### Some feasible results

##### Original image

![bungee](images/input.jpg)

##### Mask

![bungeemask](masks/input-mask.bmp)

##### Inpainted images

Patch size 9, Gaussian smoothed with sigma=2

![bungeeresult2](results/bungee_result_9_sigma2.jpg)

##### Original image

![baby](images/baby.jpg)

##### Mask

![babymask](masks/baby-mask.bmp)

##### Inpainted image

Patch size 15, Gaussian smoothed with sigma=1

![babyresult](results/baby-result-15-sigma1.jpg)

##### Original image

![hollywood](images/hollywood.jpg)

##### Mask

![hmask](masks/hollywood-mask.bmp)

##### Inpainted image

Patch size 9, Gaussian smooth with sigma=1

![hresult](results/hollywood-result-9-sigma1.jpg)

#### Not so good results

##### Original image

![bungee](images/input.jpg)

##### Mask

![bungeemask](masks/input-mask.bmp)

##### Inpainted images

Patch size 9, Gaussian smoothed with sigma=1.625

![bungeeresult1](results/bungee_result_9_sigma1.625.jpg)

##### Original image

![golf](images/golf.jpg)

##### Mask

![golf](masks/golf-mask.pgm)

##### Inpainted image

Patch size 19, Gaussian smoothed with sigma=0.5

![golfr](results/golf_result_19_sigma0.5.jpg)

Patch size 37, no Gaussian smoothing

![golfr1](results/golf_results_37_nogauss.jpg)

##### Original image

![zoo](images/zoo.jpg)

##### Mask

![zoom](masks/zoo-mask.bmp)

##### Inpainted image

Patch size 9, Gaussian smoothed with sigma=1.55

![zoor](results/zoo-result-9-sigma1.55.jpg)

Patch size 9, Gaussian smoothed with sigma=1.625

![zoor2](results/zoo-result-9-sigma1.625.jpg)

