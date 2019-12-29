import numpy as np

from keras import layers, models, optimizers
from keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau

def lr_schedule(epoch):
    """Learning Rate Schedule
    """
    lr = 0.5e-2
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 90:
        lr *= 1e-3
    elif epoch > 60:
        lr *= 5e-2
    elif epoch > 30:
        lr *= 5e-1
    print('Learning rate: ', lr)
    return lr

class ClassifyNet:
    def __init__(self):
        self.convBase = InceptionV3(include_top=False, weights='imagenet', input_shape=(224, 224, 3), classes=2)
       # self.convBase.summary()
        self.model = models.Sequential()
        self.model.add(self.convBase)
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(2, activation='softmax'))

        for layer in self.convBase.layers:
             layer.trainable = False

             if layer.name is "mixed8":
                break

        for layer in self.convBase.layers:
            print(layer.name, ' - trainable - ', layer.trainable)

        self.model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
        

        #TODO : refactor
        # trainDatagen = ImageDataGenerator(rescale=1./255,   #Scale the image between 0 and 1
        #                             rotation_range=40,
        #                             width_shift_range=0.2,
        #                             height_shift_range=0.2,
        #                             shear_range=0.2,
        #                             zoom_range=0.2,
        #                             horizontal_flip=True,
        #                             fill_mode='nearest')

        # valDatagen = ImageDataGenerator(rescale=1./255)  #We do not augment validation data. we only perform rescale

        # Training parameters
        batch_size = 32
        epochs = 20
        num_classes = 2

        # =============== DATA loading ========================

        train_path = '/home/jakub/projects/minutiae-extractor/ClassifyNet/Dataset/train/'
        test_path = '/home/jakub/projects/minutiae-extractor/ClassifyNet/Dataset/validate/'

        input_shape = (224, 224, 3)
        
        # Using data augmentation technique for training
        datagen = ImageDataGenerator(
                # set input mean to 0 over the dataset
                featurewise_center=False,
                # set each sample mean to 0
                samplewise_center=False,
                # divide inputs by std of dataset
                featurewise_std_normalization=False,
                # divide each input by its std
                samplewise_std_normalization=False,
                # apply ZCA whitening
                zca_whitening=False,
                # randomly rotate images in the range (deg 0 to 180)
                rotation_range=180,
                # randomly shift images horizontally
                width_shift_range=0.5,
                # randomly shift images vertically
                height_shift_range=0.5,
                # randomly flip images
                horizontal_flip=True,
                # randomly flip images
                vertical_flip=True)

        train_batches = datagen.flow_from_directory(train_path, target_size=(input_shape[0], input_shape[1]), classes=['ending', 'bifurcation'], batch_size=batch_size, color_mode='rgb')
        # Feed data from directory into batches
        test_gen = ImageDataGenerator()
        test_batches = test_gen.flow_from_directory(test_path, target_size=(input_shape[0], input_shape[1]), classes=['ending', 'bifurcation'], batch_size=batch_size, color_mode='rgb')

        lr_scheduler = LearningRateScheduler(lr_schedule)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
        callbacks = [lr_reducer, lr_scheduler]

        #The training part
        #We train for 64 epochs with about 100 steps per epoch
        history = self.model.fit_generator(train_batches,
                                    epochs=epochs,
                                    validation_data=test_batches,
                                    verbose=1,
                                    callbacks=callbacks)

        score = self.model.evaluate_generator(test_batches)
        print('Test accuracy:', score[1])