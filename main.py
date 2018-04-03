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
os.environ['CUDA_VISIBLE_DEVICES']='0'

def step_decay(epoch, lr=0.1):
	initial_lrate = lr
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

# get data generator
img_rows = 384
img_cols = 1248
train_image_path = '/home/xinyang/Documents/roadSeg/data/data_road/training/processed_image'
train_mask_path = '/home/xinyang/Documents/roadSeg/data/data_road/training/processed_mask'
train_batch_size = 2
val_image_path = '/home/xinyang/Documents/roadSeg/data/data_road/training/val_image'
val_mask_path = '/home/xinyang/Documents/roadSeg/data/data_road/training/val_mask'
val_batch_size = 1
epochs=100
lr=0.00001
input_shape=(img_rows, img_cols, 3)
log_save_path = 'run_logs/run4.csv'
weight_save_path = 'weights/bestWeights_run4.hdf5'

data_train_gen_args = dict(width_shift_range=0.2,
                     height_shift_range=0.2,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     rescale=1./255)

train_generator = trainGenerate(img_path=train_image_path, 
                                msk_path=train_mask_path,
                                gen_args=data_train_gen_args,
                                batch_size=train_batch_size,
                                imgSize=(img_rows, img_cols))

data_val_gen_args = dict(rescale=1./255)
val_generator = valGenerate(img_path= val_image_path,
                            msk_path = val_mask_path,
                            gen_args = data_val_gen_args,
                            batch_size = val_batch_size,
                            imgSize=(img_rows, img_cols))


######################## get Model ready
model = getUNet(input_shape=input_shape,
                lr=lr,
                loss= bce_dice_loss,
                metrics=[dice_coeff],
                num_classes=1)

callbacks = [LearningRateScheduler(partial(step_decay, lr=lr)),
             CSVLogger(log_save_path, append=True, separator=';'),
             ModelCheckpoint(monitor = 'val_loss',
                             filepath=weight_save_path,
                             save_best_only = True,
                             save_weights_only = True)]

model.fit_generator(generator=train_generator,
                    epochs = epochs,
                    steps_per_epoch=int(244/train_batch_size),
                    callbacks=callbacks,
                    validation_data=val_generator,
                    validation_steps=int(30/val_batch_size))

