#save and load models
from keras.models import load_model
model.save('partly_trained_run4.h5')

model = load_model('partly_trained_run4.h5')

# then fit the model as usual

model.fit_generator(generator=train_generator,
                    epochs = epochs,
                    steps_per_epoch=int(244/train_batch_size),
                    callbacks=callbacks,
                    validation_data=val_generator,
                    validation_steps=int(30/val_batch_size))