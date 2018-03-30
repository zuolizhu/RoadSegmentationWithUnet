from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
############################## Get data generator ready
# input image size
img_rows = 375
img_cols = 1242
image_path = '/home/xinyang/Documents/roadSeg/data/data_road/training/processed_image'
mask_path = '/home/xinyang/Documents/roadSeg/data/data_road/training/processed_mask'
batch_size = 16
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     validation_split=0.2,
                     rescale=1./255)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
#image_datagen.fit(images, augment=True, seed=seed)
#mask_datagen.fit(masks, augment=True, seed=seed)

image_generator = image_datagen.flow_from_directory(
    directory = image_path,
    batch_size = batch_size,
    target_size=(img_rows, img_cols),
     color_mode='rgb',
    class_mode=None,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    mask_path,
    batch_size = batch_size,
    target_size=(img_rows, img_cols),
    color_mode='grayscale',
    class_mode=None,
    seed=seed)
train_generator = zip(image_generator, mask_generator)

######################### test generator (not done yet)
'''testImg_datagen = ImageDataGenerator(featurewise_center=True,
                     featurewise_std_normalization=True)
testMsk_datagen = ImageDataGenerator(featurewise_center=True,
                     featurewise_std_normalization=True)
testImg_generator = image_datagen.flow_from_directory(
    'data/images',
    class_mode=None,
    seed=seed)
testMsk_generator = image_datagen.flow_from_directory(
    'data/images',
    class_mode=None,
    seed=seed)'''
########################
############################### end of data generator

######################## get Model ready
from RoadSeg_model import unet1
# combine generators into one which yields image and masks

model=unet1(img_rows, img_cols)
model.compile(optimizer=Adam(lr=1e-4), 
              loss=IOU_calc_loss, metrics=[IOU_calc])
model.fit_generator(
    train_generator,
    epochs=50)