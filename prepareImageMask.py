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
    #mask1[mask1 > 1]=1
    return mask1



savePath_msk ='/home/xinyang/Documents/roadSeg/data/data_road/training/processed_mask/processed masks'
savePath_img ='/home/xinyang/Documents/roadSeg/data/data_road/training/processed_image/processed_image'
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
    img_name = df['image_name'][ind]
    img_path = df['image_path'][ind]
    mask_name = df['mask_name'][ind]
    mask_path = df['mask_path'][ind]
    
    img = cv2.imread(img_path,1)
    img_size=np.shape(img)
    if size != img_size[0:2]:
        img = padImg(img, size);
    
    msk = cv2.imread(mask_path,1)    
    msk_size = np.shape(msk)
    if size != msk_size[0:2]:
        msk = padImg(msk, size);
        
    return img_name, img, mask_name, msk
#################################################

'''Process the images and masks.
images and masks are not all in the same size, so read the images and resize them in
getImageMask function. The masks are 3 channel and only the first channel records the road,
therefore, obtain the first channel (pixle value range 0-255) in processMask() function.
and export the images and masks into image_path, and mask_path in the following function.
'''

########## main function

image_path = '/home/xinyang/Documents/roadSeg/data/data_road/training/original_image'
mask_path = '/home/xinyang/Documents/roadSeg/data/data_road/training/original_masks'
images = sorted(glob.glob(os.path.join(image_path, '*.png')))
masks = sorted(glob.glob(os.path.join(mask_path, '*road*.png')))

image_names = list(map(lambda x: os.path.basename(x), images))
mask_names = list(map(lambda x: os.path.basename(x), masks))

name_dict = {'image_name': image_names, 'image_path':images, 
             'mask_name': mask_names, 'mask_path':masks}
df = pd.DataFrame(name_dict)

for idx in range(df.shape[0]):
    img_name, img, msk_name, msk = getImageMask(df, idx, size=(384, 1248))
    msk2= processMask(msk)
    cv2.imwrite(os.path.join(savePath_msk, msk_name), msk2)
    cv2.imwrite(os.path.join(savePath_img, img_name), img)

