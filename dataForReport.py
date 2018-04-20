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
from postProcessing import getAveIOU, predPostPorcess, predBatchPostProcess
import os 
from helper import plotImgMsk, showImage, plotPredTruth
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
threshold = 0.45

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

######### Get predictions from images
images = test_generator[1]
truth = test_truth_generator[1]
predictions = model.predict(images)
predictions = predBatchPostProcess(predictions, threshold=threshold)

idx = 0
# plot some images
plotImgMsk(images[idx], truth[idx])
plotImgMsk(images[idx], predictions[idx])
showImage(images[idx], truth[idx])
plotPredTruth(truth[idx], predictions[idx])
