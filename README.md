# Image Inpainting

Implementaion of exemplar-based image inpainting algorithm by Criminisi et al. 

#### Some feasible results

##### Original image

![bungee](input.jpg)

##### Mask

![bungeemask](input-mask.bmp)

##### Inpainted images

Patch size 9, Gaussian smoothed with sigma=1.625

![bungeeresult1](results/bungee_result_9_sigma1.625.jpg)

Patch size 9, Gaussian smoothed with sigma=2

![bungeeresult2](results/bungee_result_9_sigma2.jpg)

##### Original image

![baby](baby.jpg)

##### Mask

![babymask](baby-mask.bmp)

##### Inpainted image

Patch size 15, Gaussian smoothed with sigma=1

![babyresult](results/baby-result-15-sigma1.jpg)

##### Original image

![hollywood](hollywood.jpg)

##### Mask

![hmask](hollywood-mask.bmp)

##### Inpainted image

Patch size 9, Gaussian smooth with sigma=1

![hresult](results/hollywood-result-9-sigma1.jpg)

#### Not so good results

##### Original image

![golf](golf.jpg)

##### Mask

![golf](golf-mask.pgm)

##### Inpainted image

Patch size 19, Gaussian smoothed with sigma=0.5

![golfr](results/golf_result_19_sigma0.5.jpg)

Patch size 37, no Gaussian smoothing

![golfr1](results/golf_results_37_nogauss.jpg)

##### Original image

![zoo](zoo.jpg)

##### Mask

![zoom](zoo-mask.bmp)

##### Inpainted image

Patch size 9, Gaussian smoothed with sigma=1.55

![zoor](results/zoo-result-9-sigma1.55.jpg)

Patch size 9, Gaussian smoothed with sigma=1.625

![zoor2](results/zoo-result-9-sigma1.625.jpg)

