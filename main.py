from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from generator import trainGenerate
from unetModel import getUNet
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import RMSprop
from functools import partial
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
image_path = '/home/xinyang/Documents/roadSeg/data/data_road/training/processed_image'
mask_path = '/home/xinyang/Documents/roadSeg/data/data_road/training/processed_mask'
batch_size = 4
epochs=100
lr=0.001
input_shape=(img_rows, img_cols, 3)

data_gen_args = dict(width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     validation_split=0.2,
                     rescale=1./255)

train_generator = trainGenerate(img_path=image_path, 
                                msk_path=mask_path,
                                gen_args=data_gen_args,
                                batch_size=batch_size,
                                imgSize=(img_rows, img_cols))


######################## get Model ready
model = getUNet(input_shape=input_shape,
                lr=lr,
                loss='binary_crossentropy',
                metrics=['accuracy'],
                num_classes=1)

callbacks = [LearningRateScheduler(partial(step_decay, lr=lr)),
             ModelCheckpoint(monitor = 'val_loss',
                             filepath='weights/best_weights.hdf5',
                             save_best_only = True,
                             save_weights_only = True)]

model.fit_generator(generator=train_generator,
                    epochs = epochs,
                    steps_per_epoch=int(289*0.8/batch_size),
                    callbacks=callbacks)

