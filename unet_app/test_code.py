# Test
from keras import backend as K
image = images
y_true = ground_truth
y_pred = predictions 


y_true = K.cast(y_true, 'float32')
y_pred = K.cast(y_pred, 'float32')
# if we want to get same size of output, kernel size must be odd number
kernel_size = 41
averaged_mask = K.pool2d(y_true, pool_size=(kernel_size, kernel_size), strides=(1, 1), padding='same', pool_mode='avg')

border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
weight = K.ones_like(averaged_mask)
w0 = K.sum(weight)
weight += border * 2
w1 = K.sum(weight)
weight *= (w0 / w1)
    
smooth = 1.
w, m1, m2 = weight * weight, y_true, y_pred
intersection = (m1 * m2)
score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)