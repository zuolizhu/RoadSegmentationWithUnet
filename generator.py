'''
generator.py contains data generators, including train, validation and test generator
'''

from keras.preprocessing.image import ImageDataGenerator

########################### tain generator
def trainGenerate(img_path, msk_path, gen_args, batch_size=16, imgSize=(384, 1248)):
    '''
    train data generator.
    Input:
        img_path -- path for getting training images.
        msk_path -- path for ground truth masks
        gen_args -- argumetns for train generator
        batch_size -- batch size 
        imgSize -- image size after generated
    output:
        train_generator -- data generator for training
    '''
    img_rows = imgSize[0]
    img_cols = imgSize[1]
    image_path = img_path
    mask_path = msk_path
    batch_size = batch_size
    data_gen_args = gen_args
    image_datagen = ImageDataGenerator(**data_gen_args)
    
    
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    
    
    seed = 1
    image_generator = image_datagen.flow_from_directory(
    directory = image_path,
    batch_size = batch_size,
    target_size=(img_rows, img_cols),
    color_mode='rgb',
    class_mode=None,
    shuffle=True,
    seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
    mask_path,
    batch_size = batch_size,
    target_size=(img_rows, img_cols),
    color_mode='grayscale',
    class_mode=None,
    shuffle=True,
    seed=seed)
    
    train_generator = zip(image_generator, mask_generator)
    return train_generator
    #return image_generator, mask_generator


############################ validation generator
def valGenerate(img_path, msk_path, gen_args, batch_size=8, imgSize=(384, 1248)):
    '''
    Validation data generator.
    Input:
        img_path -- path for getting training images.
        msk_path -- path for ground truth masks
        gen_args -- argumetns for train generator
        batch_size -- batch size 
        imgSize -- image size after generated
    output:
        val_generator -- data generator for validation
    '''
    img_rows = imgSize[0]
    img_cols = imgSize[1]
    image_path = img_path
    mask_path = msk_path
    data_gen_args = gen_args
    image_datagen = ImageDataGenerator(**data_gen_args)    
    
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    seed = 2
    image_generator = image_datagen.flow_from_directory(
    directory = image_path,
    batch_size = batch_size,
    target_size=(img_rows, img_cols),
    color_mode='rgb',
    class_mode=None,
    shuffle=True,
    seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
    mask_path,
    batch_size = batch_size,
    target_size=(img_rows, img_cols),
    color_mode='grayscale',
    class_mode=None,
    shuffle=True,
    seed=seed)
    
    val_generator = zip(image_generator, mask_generator)
    return val_generator
######################### test generator (not done yet)


def testDataGenerate(img_path, gen_args, batch_size=8, imgSize=(384, 1248), seed=0):
    '''
    Test data generator. Only retrieve the RGB images. 
    Input:
        img_path -- path for getting training images.
        msk_path -- path for ground truth masks
        gen_args -- argumetns for train generator
        batch_size -- batch size 
        imgSize -- image size after generated
        seed -- keep the seed for testDataGenerate and testTruthGenerate to 
                ensure the same order of masks and images
    output:
        image_generator -- data generator for test images
    '''
    img_rows = imgSize[0]
    img_cols = imgSize[1]
    image_path = img_path

    data_gen_args = gen_args
    
    image_datagen = ImageDataGenerator(**data_gen_args) 

    seed = seed
    image_generator = image_datagen.flow_from_directory(
    directory = image_path,
    batch_size = batch_size,
    target_size=(img_rows, img_cols),
    color_mode='rgb',
    class_mode=None,
    shuffle=True,
    seed=seed)
    return image_generator
########################
def testTruthGenerate(img_path, gen_args, batch_size=8, imgSize=(384, 1248), see=0):
    '''
    Test data generator. Only retrieve the RGB images. 
    Input:
        img_path -- path for getting training images.
        msk_path -- path for ground truth masks
        gen_args -- argumetns for train generator
        batch_size -- batch size 
        imgSize -- image size after generated
                seed -- keep the seed for testDataGenerate and testTruthGenerate to 
                ensure the same order of masks and images
    output:
        test_generator -- data generator for test masks
    '''
    img_rows = imgSize[0]
    img_cols = imgSize[1]
    image_path = img_path

    data_gen_args = gen_args
    
    image_datagen = ImageDataGenerator(**data_gen_args) 

    seed = seed
    image_generator = image_datagen.flow_from_directory(
    directory = image_path,
    batch_size = batch_size,
    target_size=(img_rows, img_cols),
    color_mode='grayscale',
    class_mode=None,
    shuffle=True,
    seed=seed)
    return image_generator