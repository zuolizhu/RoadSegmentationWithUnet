# Helper
import cv2
from skimage import data, color, io, img_as_float
import numpy as np
import matplotlib.pyplot as plt

def showImage(img1, img2):
    cv2.imshow('image',img1)
    cv2.imshow('image2', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plotImgMsk(img, msk, threshold=0.5):      
    alpha=0.6
    color_mask = np.zeros(img.shape)
    msk[msk>=threshold]=1
    msk[msk<threshold]=0
    color_mask[:,:,0]=msk[:,:,0]
    
    # Convert the input image and color mask to Hue Saturation Value (HSV)
    # colorspace
    img_hsv = color.rgb2hsv(img)
    color_mask_hsv = color.rgb2hsv(color_mask)
    
    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    # Display the output
    
    #ax0.imshow(img, cmap=plt.cm.gray) 
    #ax1.imshow(color_mask)
    plt.imshow(img_masked)
    #plt.figure()
    plt.show()

def plotPredTruth(truth, pred, threshold=0.5):
    alpha=0.5

    color_pred = np.zeros(truth.shape[:2]+ (3,))
    pred[pred>=threshold]=1
    pred[pred<threshold]=0
    color_pred[:,:,2] = pred[:,:,0]
    
    truth2=np.copy(truth)
    truth2[truth2==1] = 2
    truth2[truth2==0] = 1
    truth2[truth2==2] = 0
    color_truth = np.ones(truth.shape[:2]+ (3,))
    color_truth[:,:,1] = truth2[:,:,0]
    
    dst = cv2.addWeighted(color_truth,0.7,color_pred,0.7,0)
    cv2.imshow('dst',dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    '''color_truth_hsv = color.rgb2hsv(color_truth)
    color_pred_hsv = color.rgb2hsv(color_pred)
    
    color_truth_hsv[..., 0] = color_pred_hsv[..., 0] * alpha
    color_truth_hsv[..., 1] = color_pred_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(color_truth_hsv)
    # Display the output
    
    #ax0.imshow(img, cmap=plt.cm.gray)
    #ax1.imshow(color_mask)
    plt.imshow(img_masked)
    #plt.figure()
    plt.show()
    return color_truth, color_pred'''