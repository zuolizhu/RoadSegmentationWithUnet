from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger
from generator import trainGenerate, valGenerate
from unetModel import getUNet, getThinnerUNet, getThinnerUNet5Pool
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import RMSprop
from functools import partial
from lossFunction import dice_coeff, dice_loss, bce_dice_loss, weighted_dice_coeff, weighted_dice_loss, weighted_bce_dice_loss
from lossFunction import my_dice_loss, my_bce_loss
import math
import os 
from keras.models import load_model
os.environ['CUDA_VISIBLE_DEVICES']='0'

# Weight decay function
def step_decay(epoch, lr=0.1):
	initial_lrate = lr
	drop = 0.5
	epochs_drop = 20.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

# input parameters
img_rows = 384
img_cols = 1248
train_image_path = '/home/xinyang/Documents/roadSeg/data/data_road/training/processed_image'
train_mask_path = '/home/xinyang/Documents/roadSeg/data/data_road/training/processed_mask'
train_batch_size = 4
val_image_path = '/home/xinyang/Documents/roadSeg/data/data_road/training/val_image'
val_mask_path = '/home/xinyang/Documents/roadSeg/data/data_road/training/val_mask'
val_batch_size = 1
epochs=200
lr=0.0001
input_shape=(img_rows, img_cols, 3)
exp_name = 'run14'
log_save_path = 'run_logs/' + exp_name + '.csv'
weight_save_path = 'weights/' + exp_name + '.hdf5'
model_save_path = 'models/' + exp_name + '.h5'

# get training data generator
data_train_gen_args = dict(width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.1,
                     fill_mode='constant', 
                     cval=0.0,
                     horizontal_flip=True,
                     rescale=1./255)

train_generator = trainGenerate(img_path=train_image_path, 
                                msk_path=train_mask_path,
                                gen_args=data_train_gen_args,
                                batch_size=train_batch_size,
                                imgSize=(img_rows, img_cols))

# get validation data generator
data_val_gen_args = dict(rescale=1./255)
val_generator = valGenerate(img_path= val_image_path,
                            msk_path = val_mask_path,
                            gen_args = data_val_gen_args,
                            batch_size = val_batch_size,
                            imgSize=(img_rows, img_cols))


######################## get Model ready
model = getThinnerUNet(input_shape=input_shape,
                lr=lr,
                loss= weighted_bce_dice_loss,
                metrics=[dice_coeff, my_bce_loss, my_dice_loss],
                num_classes=1)

callbacks = [LearningRateScheduler(partial(step_decay, lr=lr)),
             CSVLogger(log_save_path, append=True, separator=';'),
             ModelCheckpoint(monitor = 'val_loss',
                             filepath=weight_save_path,
                             save_best_only = True,
                             save_weights_only = True)]

# Train model
model.fit_generator(generator=train_generator,
                    epochs = epochs,
                    steps_per_epoch=int(244/train_batch_size),
                    callbacks=callbacks,
                    validation_data=val_generator,
                    validation_steps=int(30/val_batch_size))

# Save model
model.save(model_save_path)
