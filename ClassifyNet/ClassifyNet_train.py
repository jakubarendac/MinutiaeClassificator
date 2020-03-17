
"""Code for FineNet in paper "Robust Minutiae Extractor: Integrating Deep Networks and Fingerprint Domain Knowledge" at ICB 2018
  https://arxiv.org/pdf/1712.09401.pdf

  If you use whole or partial function in this code, please cite paper:

  @inproceedings{Nguyen_MinutiaeNet,
    author    = {Dinh-Luan Nguyen and Kai Cao and Anil K. Jain},
    title     = {Robust Minutiae Extractor: Integrating Deep Networks and Fingerprint Domain Knowledge},
    booktitle = {The 11th International Conference on Biometrics, 2018},
    year      = {2018},
    }
"""

import math
import os
import sys
from datetime import datetime

import numpy as np
from keras.callbacks import (LearningRateScheduler, ModelCheckpoint,
                             ReduceLROnPlateau, TensorBoard)
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix

from MinutiaeClassificator.ClassifyNet.ClassifyNet_constants import MINUTIAE_CLASSES
from MinutiaeClassificator.ClassifyNet.ClassifyNet_model import ClassifyNetModel
from MinutiaeClassificator.MinutiaeNet.FineNet.FineNet_model import plot_confusion_matrix


output_dir = './output_ClassifyNet/'+datetime.now().strftime('%Y%m%d-%H%M%S')

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), output_dir)
log_dir = os.path.join(os.getcwd(), output_dir + '/logs')

# TODO : before training adjust training parameters

# Training parameters
batch_size = 32
epochs = 100
num_classes = 4
train_data_count = 40000
validation_data_count = 4000

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# Model size, patch
model_type = 'patch224batch32'


# =============== DATA loading ========================

# TODO : before training adjust data paths

train_path = '/home/jakub/projects/Dataset/train/'
test_path = '/home/jakub/projects/Dataset/validate/'

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

train_batches = datagen.flow_from_directory(train_path, target_size=(
    input_shape[0], input_shape[1]), classes=MINUTIAE_CLASSES, batch_size=batch_size)
# Feed data from directory into batches
test_gen = ImageDataGenerator()
test_batches = test_gen.flow_from_directory(test_path, target_size=(
    input_shape[0], input_shape[1]), classes=MINUTIAE_CLASSES, batch_size=batch_size)


# =============== end DATA loading ========================


def lr_schedule(epoch):
    """Learning Rate Schedule
    """
    lr = 0.5e-2
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 150:
        lr *= 1e-3
    elif epoch > 60:
        lr *= 5e-2
    elif epoch > 30:
        lr *= 5e-1
    print('Learning rate: ', lr)
    return lr


# ============== Define model ==================

# TODO : before training adjust pretrained path of network

model = ClassifyNetModel(num_classes=num_classes,
                         pretrained_path='../MinutiaeNet/Models/FineNet.h5',
                         input_shape=input_shape,
                         load_layers_by_name=True)

# Save model architecture
#plot_model(model, to_file='./modelClassifyNet.pdf',show_shapes=True)

# best trainings for 2 classes:
# 92% - mixed_7a - 50 epochs - test acc: 89%
# 94% - mixed_6a - 50 epochs - test acc: 91%
# 93,5% - mixed_6a - 100 epochs - test acc: 92,5%

# best trainings for 4 classes:
#

# Freeze not trainable layers
for layer in model.layers:
    layer.trainable = False

    if layer.name is "mixed_6a":
        break

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
print model.summary()

# ============== End define model ==============

# ============== Other stuffs for loging and parameters ==================
model_name = 'ClassifyNet_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

filepath = os.path.join(save_dir, model_name)


# Show in tensorboard
tensorboard = TensorBoard(
    log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler, tensorboard]

# ============== End other stuffs  ==================

# Begin training
model.fit_generator(train_batches,
                    steps_per_epoch=math.ceil(train_data_count / batch_size),
                    validation_data=test_batches,
                    validation_steps=math.ceil(
                        validation_data_count / batch_size),
                    epochs=epochs, verbose=1,
                    callbacks=callbacks)


# Plot confusion matrix
score = model.evaluate_generator(test_batches)
print 'Test accuracy:', score[1]
predictions = model.predict_generator(test_batches)
test_labels = test_batches.classes[test_batches.index_array]

cm = confusion_matrix(test_labels, np.argmax(predictions, axis=1))
plot_confusion_matrix(cm, MINUTIAE_CLASSES, title='Confusion Matrix')
