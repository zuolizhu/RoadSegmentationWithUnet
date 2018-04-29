#canny results
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
from postProcessing import getAveIOU, getAveROC, IOUcalc
import os 
from helper import plotImgMsk
import numpy as np
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES']='0'

test_image_path = '/home/xinyang/Documents/roadSeg/data/data_road/training/Canny_valid'
test_truth_path = '/home/xinyang/Documents/roadSeg/data/data_road/training/val_mask'

img_rows = 384
img_cols = 1248
input_shape=(img_rows, img_cols, 3)
lr=0.0001
batch_size = 15


data_test_gen_args = dict(rescale=1./255)
test_generator = testTruthGenerate(test_image_path,
                              gen_args=data_test_gen_args,
                              batch_size=batch_size,
                              imgSize=(img_rows, img_cols))

test_truth_generator = testTruthGenerate(test_truth_path,
                              gen_args=data_test_gen_args,
                              batch_size=batch_size,
                              imgSize=(img_rows, img_cols))

IOUscore = []
while True:
    pred = test_generator[0]
    truth = test_truth_generator[0]
    for predImg, msk in zip(pred, truth):
        IOUscore.append(IOUcalc(msk, predImg))    
    test_generator.next()
    test_truth_generator.next()
    if test_generator.batch_index==0:
        break
print(sum(IOUscore)/len(IOUscore))

