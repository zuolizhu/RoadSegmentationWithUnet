# finalAnalysis
from generator import testDataGenerate, testTruthGenerate
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger
from generator import trainGenerate, valGenerate
from unetModel import getUNet
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import RMSprop
from functools import partial
from lossFunction import dice_coeff, dice_loss, bce_dice_loss
import math
import os 
from helper import plotImgMsk


test_image_path = '/home/xinyang/Documents/roadSeg/data/data_road/training/val_image'
test_truth_path = '/home/xinyang/Documents/roadSeg/data/data_road/training/val_mask'

img_rows = 384
img_cols = 1248
input_shape=(img_rows, img_cols, 3)
lr=0.0001



model = getUNet(input_shape=input_shape,
                lr=lr,
                loss= bce_dice_loss,
                metrics=[dice_coeff],
                num_classes=1)
model.load_weights('weights/bestWeights_run3.hdf5')


data_test_gen_args = dict(rescale=1./255)
test_generator = testDataGenerate(test_image_path,
                              gen_args=data_test_gen_args,
                              batch_size=2,
                              imgSize=(img_rows, img_cols))
test_truth_generator = testTruthGenerate(test_truth_path,
                              gen_args=data_test_gen_args,
                              batch_size=2,
                              imgSize=(img_rows, img_cols))

images = test_generator.next()
ground_truth = test_truth_generator.next()
predictions = model.predict(images)



