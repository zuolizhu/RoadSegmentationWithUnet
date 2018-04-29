import cv2
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
################### Pad images
def padImg(img, size=(384, 1248)):
    if len(img.shape)==3: 
        newImg = np.zeros(list(size)+[3], dtype=np.uint8)
        newImg[4:4+img.shape[0],4:4+img.shape[1],:] = img
    else: 
        newImg = np.zeros(size, dtype=np.unit8)
        newImg[4:4+img.shape[0],4:img.shape[1]] = img
    return newImg


#######################
#Process mask, original mask are rgb image, needs to convert it into 1 channel
def processMask(mask):
    '''
    convert RGB image to masks, KITTI dataset only
    Input:
    ------
    mask: the RGB mask image
    
    Output:
    mask1: 1 dimensional mask with 0 and 255. 255--road, 0--non-road.
    '''
    mask1 = mask[:,:,0]
    newMask = np.zeros(list(mask1.size), dtype=np.uint8)
    newMask[mask1==7] = int(255)
    return mask1




### pre_process the images

############################ get images and masks from files and resize it to target
def getImageMask(df, ind, size=(384, 1248)):
    '''get the name and picture from paths
    Input
    ----------
        df: dataframe containing image files names and paths
        ind: index of images and corresponding masks to obtain
        size(375, 1242): desired size of images/masks, default is set as kitti images
    
    Returns:
    ----------
        img_name: name of the image
        img: the image
        mask_name: name of the mask corresponding to the image
        msk: mask'''

    mask_name = df['mask_name'][ind]
    mask_path = df['mask_path'][ind]
    
    msk = cv2.imread(mask_path,1)    
    msk_size = np.shape(msk)
    if size != msk_size[0:2]:
        msk = padImg(msk, size);
        
    return mask_name, msk
#################################################

'''Process the images and masks.
images and masks are not all in the same size, so read the images and resize them in
getImageMask function. The masks are 3 channel and only the first channel records the road,
therefore, obtain the first channel (pixle value range 0-255) in processMask() function.
and export the images and masks into image_path, and mask_path in the following function.
'''

########## main function
mask_path = '/home/xinyang/Documents/roadSeg/data/city_scape/cityscape/masks/train_mask'
savePath_msk ='/home/xinyang/Documents/roadSeg/data/city_scape/cityscape/processed_masks/train_masks'


masks = sorted(glob.glob(os.path.join(mask_path, '*.png')))

mask_names = list(map(lambda x: os.path.basename(x), masks))

name_dict = {'mask_name': mask_names, 'mask_path':masks}
df = pd.DataFrame(name_dict)

for idx in range(df.shape[0]):
    msk_name, msk = getImageMask(df, idx, size=(384, 1248))
    msk2= processMask(msk)
    cv2.imwrite(os.path.join(savePath_msk, msk_name), msk2)

