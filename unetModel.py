'''
unetModel.py contains 3 different Unet architectures. 
'''
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization
from keras.optimizers import RMSprop, Adam, Nadam
from keras.initializers import TruncatedNormal
from keras import regularizers
from keras.utils import multi_gpu_model
import tensorflow as tf
# import different losses

def getUNet(input_shape=(384, 1248, 3), num_classes=1, lr=0.0001, loss='categorical_crossentropy',
            metrics=['acc']):
    '''
    Set up the architecture for the UNet convolutional neural network for image segmentation. First 
    convolutional layer has 64 channels.
    Input:
        input_shape -- size of image (rows, coloumns, channels)
        num_classes -- the number of classes of the image
        lr -- learning rate
        loss -- loss function used for training
        metrics -- metrics to evaluate model
    Output:
        model
    '''
    inputs = Input(shape=input_shape)
    # 1248X384X3
    init = TruncatedNormal(mean=0., stddev=0.01, seed = None)  # convolutional layer kernel initializer
    regularizer = regularizers.l2(0.0001) # convolutional layer kernel regulizer
    down1 = Conv2D(64, (3,3), padding='same', kernel_initializer=init, kernel_regularizer=regularizer)(inputs)
    #down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3,3), padding='same', kernel_initializer=init, kernel_regularizer=regularizer)(down1)
    #down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    # 1248X384X64
    down1_pool = MaxPooling2D(pool_size=(2,2), strides=(2,2))(down1)
    # 624X192X64
    
    down2 = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_regularizer=regularizer)(down1_pool)
    #down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_regularizer=regularizer)(down2)
    #down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    # 624X192X128
    down2_pool = MaxPooling2D(pool_size=(2,2), strides=(2,2))(down2)
    # 312X96X128
    
    down3 = Conv2D(256, (3,3), padding='same', kernel_initializer=init, kernel_regularizer=regularizer)(down2_pool)
    #down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3,3), padding='same', kernel_initializer=init, kernel_regularizer=regularizer)(down3)
    #down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    # 312X96X256
    down3_pool = MaxPooling2D(pool_size=(2,2), strides=(2,2))(down3)
    # 156X48X256
    
    down4 = Conv2D(512, (3,3), padding='same', kernel_initializer=init, kernel_regularizer=regularizer)(down3_pool)
    #down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3,3), padding='same', kernel_initializer=init, kernel_regularizer=regularizer)(down4)
    #down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    # 156X48X512
    down4_pool = MaxPooling2D(pool_size=(2,2), strides=(2,2))(down4)
    # 78X24X512
    
    center = Conv2D(1024, (3,3), padding='same', kernel_initializer=init, kernel_regularizer=regularizer)(down4_pool)
    #center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3,3), padding='same', kernel_initializer=init, kernel_regularizer=regularizer)(center)
    #center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # Center
    # 78X24X1024
    
    up4 = UpSampling2D(size=(2,2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3,3), padding='same', kernel_initializer=init, kernel_regularizer=regularizer)(up4)
    #up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3,3), padding='same', kernel_initializer=init, kernel_regularizer=regularizer)(up4)
    #up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # up4  
    # 156X48X512
    
    up3 = UpSampling2D(size=(2,2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3,3), padding='same', kernel_initializer=init, kernel_regularizer=regularizer)(up3)
    #up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3,3), padding='same', kernel_initializer=init, kernel_regularizer=regularizer)(up3)
    #up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # up3 312X96X256
    
    up2 = UpSampling2D(size=(2,2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_regularizer=regularizer)(up2)
    #up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_regularizer=regularizer)(up2)
    #up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # up2 624X192X128
    
    up1 = UpSampling2D(size=(2,2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3,3), padding='same', kernel_initializer=init, kernel_regularizer=regularizer)(up1)
    #up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3,3), padding='same', kernel_initializer=init, kernel_regularizer=regularizer)(up1)
    #up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # up1 1248X384X64
    
    classify = Conv2D(num_classes, (1,1), activation='sigmoid')(up1)
    
    model = Model(inputs = inputs, outputs = classify)
    model.compile(optimizer=Nadam(lr=lr), loss = loss, metrics=metrics)
    return model
    

    
def getThinnerUNet(input_shape=(384, 1248, 3), num_classes=1, lr=0.0001, loss='categorical_crossentropy',
            metrics=['acc']):
    '''
    Set up the architecture for the UNet convolutional neural network for image segmentation. First 
    convolutional layer has 32 channels.
    Input:
        input_shape -- size of image (rows, coloumns, channels)
        num_classes -- the number of classes of the image
        lr -- learning rate
        loss -- loss function used for training
        metrics -- metrics to evaluate model
    Output:
        model
    '''
    inputs = Input(shape=input_shape)
    # 1248X384X3
    init = TruncatedNormal(mean=0., stddev=0.01, seed = None)   # convolutional layer kernel initializer
    regularizer = regularizers.l2(0.0001) # convolutional layer kernel regulizer
    down1 = Conv2D(32, (3,3), padding='same', kernel_initializer=init)(inputs)
    #down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(32, (3,3), padding='same', kernel_initializer=init)(down1)
    #down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    # 1248X384X64
    down1_pool = MaxPooling2D(pool_size=(2,2), strides=(2,2))(down1)
    # 624X192X64
    
    down2 = Conv2D(64, (3,3), padding='same', kernel_initializer=init)(down1_pool)
    #down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(64, (3,3), padding='same', kernel_initializer=init)(down2)
    #down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    # 624X192X128
    down2_pool = MaxPooling2D(pool_size=(2,2), strides=(2,2))(down2)
    # 312X96X128
    
    down3 = Conv2D(128, (3,3), padding='same', kernel_initializer=init)(down2_pool)
    #down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(128, (3,3), padding='same', kernel_initializer=init)(down3)
    #down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    # 312X96X256
    down3_pool = MaxPooling2D(pool_size=(2,2), strides=(2,2))(down3)
    # 156X48X256
    
    down4 = Conv2D(128, (3,3), padding='same', kernel_initializer=init)(down3_pool)
    #down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(128, (3,3), padding='same', kernel_initializer=init)(down4)
    #down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    # 156X48X512
    down4_pool = MaxPooling2D(pool_size=(2,2), strides=(2,2))(down4)
    # 78X24X512
    
    center = Conv2D(512, (3,3), padding='same', kernel_initializer=init)(down4_pool)
    #center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(518, (3,3), padding='same', kernel_initializer=init)(center)
    #center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # Center
    # 78X24X1024
    
    up4 = UpSampling2D(size=(2,2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(256, (3,3), padding='same', kernel_initializer=init)(up4)
    #up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(256, (3,3), padding='same', kernel_initializer=init)(up4)
    #up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # up4  
    # 156X48X512
    
    up3 = UpSampling2D(size=(2,2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(128, (3,3), padding='same', kernel_initializer=init)(up3)
    #up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(128, (3,3), padding='same', kernel_initializer=init)(up3)
    #up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # up3 312X96X256
    
    up2 = UpSampling2D(size=(2,2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(64, (3,3), padding='same', kernel_initializer=init)(up2)
    #up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(64, (3,3), padding='same', kernel_initializer=init)(up2)
    #up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # up2 624X192X128
    
    up1 = UpSampling2D(size=(2,2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(32, (3,3), padding='same', kernel_initializer=init)(up1)
    #up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(32, (3,3), padding='same', kernel_initializer=init)(up1)
    #up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # up1 1248X384X64 
        
    classify = Conv2D(num_classes, (1,1), activation='sigmoid')(up1)
    
    model = Model(inputs = inputs, outputs = classify)
    
    model.compile(optimizer=Nadam(lr=lr), loss = loss, metrics=metrics)
    return model
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    