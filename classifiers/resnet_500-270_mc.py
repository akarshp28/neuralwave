from tensorflow import keras
from keras.utils import multi_gpu_model
from keras.initializers import TruncatedNormal
from keras.engine.topology import Layer
from keras.layers import Lambda
from keras import backend as K
from keras.models import Model
from keras import optimizers
from keras import layers
import numpy as np
import argparse
import pickle
import h5py
import math
import sys
import os

def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    bn_axis = -1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    bn_axis = -1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

#******************************************************************************#
history = []
lr=1e-3
epochs=40

files = os.listdir("/users/kjakkala/neuralwave/data/CSI_30_l2")

for file in files:
	if (file.endswith(".h5")):
                hf = h5py.File("/users/kjakkala/neuralwave/data/CSI_30_l2/{}".format(file), 'r')
                X_train = np.expand_dims(hf.get('X_train'), axis=-1)
                X_test = np.expand_dims(hf.get('X_test'), axis=-1)
                y_train = np.eye(30)[hf.get('y_train')]
                y_test = np.eye(30)[hf.get('y_test')]
                hf.close()

                inputs = layers.Input(shape=(X_train.shape[1:]), name='input')

                x = layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(inputs)
                x = layers.BatchNormalization(axis=-1, name='bn_conv1')(x)
                x = layers.Activation('relu')(x)
                x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

                x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
                x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')

                x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', strides=(1, 1))
                x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')

                x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', strides=(1, 1))
                x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')

                x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', strides=(1, 1))
                x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')

                x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
                x = layers.Dense(30, activation='softmax')(x)

                model = Model(inputs=inputs, outputs=x)
                model = multi_gpu_model(model, gpus=8)

                model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr, decay=1e-2), metrics=['acc'])
                history.append(model.fit(x=X_train, y=y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=2).history)
                sys.stdout.flush()

                outfile = open("/users/kjakkala/neuralwave/data/history.pkl",'wb')
                pickle.dump(history, outfile)
                outfile.close()


                del model
