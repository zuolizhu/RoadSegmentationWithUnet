# callbacks
from keras.callbacks import ModelCheckpoint
callbacks = [ModelCheckpoint(monitor = 'val_loss',
                             filepath='weights/best_weights.hdf5',
                             save_best_only = True,
                             save_weights_only = True)]