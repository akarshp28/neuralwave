from tensorflow import keras
from keras.initializers import TruncatedNormal
from keras.utils import multi_gpu_model
from sklearn.model_selection import train_test_split
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

data_set_path = "/users/kjakkala/neuralwave/data/CSI_30_spec_20PCA_5"
lr = 0.001
epochs = 200
G = 4

def read_samples(dataset_path, endswith=".csv"):
    datapaths, labels = list(), list()
    label = 0
    classes = sorted(os.walk(dataset_path).__next__()[1])

    # List each sub-directory (the classes)
    for c in classes:
        c_dir = os.path.join(dataset_path, c)
        walk = os.walk(c_dir).__next__()
        # Add each image to the training set
        for sample in walk[2]:
            # Only keeps csv samples
            if sample.endswith(endswith):
                #tmp = np.loadtxt(open(os.path.join(c_dir, sample), "rb"), delimiter=",")
                hf = h5py.File(os.path.join(c_dir, sample), 'r')
                tmp = np.array(hf.get('data')).T
                hf.close()
                if (tmp.shape == (80, 184, 5)):
                      datapaths.append(tmp)
                      labels.append(label)
        label += 1
    return np.array(datapaths), labels
#    return np.expand_dims(datapaths, axis=-1), labels

X, y = read_samples(data_set_path, ".h5")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
y_train = np.eye(30)[y_train]
y_test = np.eye(30)[y_test]

print(X_train.shape, y_train.shape)

inputs = layers.Input(shape=(X_train.shape[1:]), name='input')

x = layers.BatchNormalization(axis=-1, name='block1_bn1')(inputs)
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

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(30, activation='softmax', name='pred')(x)

model = Model(inputs=inputs, outputs=x)
model.summary()

model = multi_gpu_model(model, gpus=G)

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr), metrics=['acc'])
model.fit(x=X_train, y=y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=2)
