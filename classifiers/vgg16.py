from tensorflow import keras
from keras.initializers import TruncatedNormal
from keras.utils import multi_gpu_model
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

data_set_path = "/users/kjakkala/neuralwave/data/CSI_30_l2_AMP_500_NO_PCA.h5"
lr = 0.001
epochs = 200
G = 8

hf = h5py.File(data_set_path, 'r')
X_train = np.expand_dims(hf.get('X_train'), axis=-1)[:, 2:-2]
X_test = np.expand_dims(hf.get('X_test'), axis=-1)[:, 2:-2]
y_train = np.eye(30)[hf.get('y_train')]
y_test = np.eye(30)[hf.get('y_test')]
hf.close()

inputs = layers.Input(shape=(X_train.shape[1:]), name='input')
x = layers.ZeroPadding2D(padding=((0,0), (1,1)))(inputs)

x = layers.BatchNormalization(axis=-1, name='block1_bn1')(x)
x = layers.Activation('relu', name='block1__relu1')(x)
x = layers.Conv2D(64, (3, 3), padding='same', name='block1_conv1')(x)
x = layers.BatchNormalization(axis=-1, name='block1_bn2')(x)
x = layers.Activation('relu', name='block1__relu2')(x)
x = layers.Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

x = layers.BatchNormalization(axis=-1, name='block2_bn1')(x)
x = layers.Activation('relu', name='block2__relu1')(x)
x = layers.Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
x = layers.BatchNormalization(axis=-1, name='block2_bn2')(x)
x = layers.Activation('relu', name='block2__relu2')(x)
x = layers.Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

x = layers.BatchNormalization(axis=-1, name='block3_bn1')(x)
x = layers.Activation('relu', name='block3__relu1')(x)
x = layers.Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
x = layers.BatchNormalization(axis=-1, name='block3_bn2')(x)
x = layers.Activation('relu', name='block3__relu2')(x)
x = layers.Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(30, activation='softmax', name='predictions')(x)

model = Model(inputs=inputs, outputs=x)
model.summary()

model = multi_gpu_model(model, gpus=G)

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr), metrics=['acc'])
model.fit(x=X_train, y=y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=2)
