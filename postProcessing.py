#Post Processing
import numpy as np
from keras import backend as K
from keras.losses import binary_crossentropy

def predPostPorcess(pred_img, threshold=0.5):
    '''convert prediction probability 0 ~ 1 into binary results 0 or 1'''
    pred_img[pred_img>=threshold] = 1.
    pred_img[pred_img<threshold] = 0.
    return pred_img

########### Calculate IOU for final results
def IOUcalc(y_true, y_pred):
    smooth = 1.
    y_true_new = y_true.flatten()
    y_pred_new = y_pred.flatten()
    intersection = np.sum(y_true_new * y_pred_new)
    score = (2. * intersection + smooth) / (np.sum(y_true_new) + np.sum(y_pred_new) + smooth)
    return score

def getAveIOU(test_generator, truth_generator, model, threshold = 0.5):
    '''get average IOU score on given model, and test_data_generator and ground_truth Generator'''
    IOUscore = []
    while True:
        pred_proba = model.predict(test_generator[0])
        pred = predPostPorcess(pred_proba, threshold=threshold)
        truth = truth_generator[0]
        for predImg, msk in zip(pred, truth):
            IOUscore.append(IOUcalc(msk, predImg))    
        test_generator.next()
        truth_generator.next()
        if test_generator.batch_index==0:
            break
    return sum(IOUscore)/len(IOUscore)

def predBatchPostProcess(pred, threshold=0.5):
    for i in range(pred.shape[0]):
        pred[i] = predPostPorcess(pred[i], threshold=threshold)
    return pred
        

########### calculate AUC for final results

