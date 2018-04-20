# finalAnalysis
from generator import testDataGenerate, testTruthGenerate
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger
from generator import trainGenerate, valGenerate
from unetModel import getUNet, getThinnerUNet, getThinnerUNet5Pool
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import RMSprop
from functools import partial
from lossFunction import dice_coeff, dice_loss, bce_dice_loss
from postProcessing import getAveIOU, getAveROC
import os 
from helper import plotImgMsk
import numpy as np
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES']='0'

test_image_path = '/home/xinyang/Documents/roadSeg/data/data_road/training/val_image'
test_truth_path = '/home/xinyang/Documents/roadSeg/data/data_road/training/val_mask'

img_rows = 384
img_cols = 1248
input_shape=(img_rows, img_cols, 3)
lr=0.0001
batch_size = 15
weight_name = 'run14'
weight_file = 'weights/' + weight_name  + '.hdf5'

model = getThinnerUNet(input_shape=input_shape,
                lr=lr,
                loss= bce_dice_loss,
                metrics=[dice_coeff],
                num_classes=1)
model.load_weights(weight_file)


data_test_gen_args = dict(rescale=1./255)
test_generator = testDataGenerate(test_image_path,
                              gen_args=data_test_gen_args,
                              batch_size=batch_size,
                              imgSize=(img_rows, img_cols))

test_truth_generator = testTruthGenerate(test_truth_path,
                              gen_args=data_test_gen_args,
                              batch_size=batch_size,
                              imgSize=(img_rows, img_cols))
''' Get predictions from images
images = test_generator.next()
ground_truth = test_truth_generator.next()
predictions = model.predict(images)'''

########## given different threshold value for image prediction, find out threshold vs IOU
threshs = np.arange(0., 1.05, 0.05)
IOUsummary = pd.DataFrame({'threshold':threshs})
for item in threshs:
    temp = getAveIOU(test_generator, test_truth_generator, model, threshold=item)
    IOUsummary.loc[IOUsummary['threshold']==item, 'IOU_score'] = temp
#IOUsummary.to_csv('run14_theshold_IOU.csv')  # save IOU vs threshold in csv file

######## given different threshold values for image prediction, find average false positive rate 
    # and true positive rate for plotting ROC curve
threshs = np.arange(0., 1.05, 0.05)
ROCsummary = pd.DataFrame({'threshold':threshs})
for item in threshs:
    tpr, fpr =  getAveROC(test_generator, test_truth_generator, model, threshold=item)
    ROCsummary.loc[ROCsummary['threshold']==item, 'tpr'] = tpr
    ROCsummary.loc[ROCsummary['threshold']==item, 'fpr'] = fpr
ROCsummary.to_csv('run14_theshold_ROC.csv') # save ROC curve in csv file
