# RoadSegmentationWithUnet
## 1. prepare images and masks for KITTI (prepareImageMask.py)
original training images and masks are not ready for training for 3 reasons:
a. Samples are RGB images, however, the sizes (row, cols) are not the same. They need to be resized
    into the same dimension.
b. Masks are RGB images with 3 channels, only the first channel records the label of road
    this preprocessing will pick the first channel, and save it as a gray image with values between
    0-255.
c. There are more masks than the samples. The masks with 2 different labels, 'road' and 'lane'. 
    We only used the one with roads for processing in step b. The code in prepareImageMask.py will
    only pick up th images labeled with 'road' in its file name for preparing the images.
Input needed are the folder of the images, masks and the ones to save processed images and masks.
Note: in getImageMask(df, ind, size=(1242, 375)), 1242 is the number of columns, and 375 are number of 
    rows in the images. 
