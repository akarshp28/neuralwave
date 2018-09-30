from tensorflow import keras
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

def identity_block_original_3l(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters

    bn_axis = -1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv1D(filters1, 1,
                      kernel_initializer='he_normal',
                      name=conv_name_base + 'a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + 'a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + 'b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + 'b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(filters3, 1,
                      kernel_initializer='he_normal',
                      name=conv_name_base + 'c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + 'c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x

def identity_block_new_3l(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters

    bn_axis = -1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + 'a')(input_tensor)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(filters1, 1,
                      kernel_initializer='he_normal',
                      name=conv_name_base + 'a')(x)

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + 'b')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + 'c')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(filters3, 1,
                      kernel_initializer='he_normal',
                      name=conv_name_base + 'c')(x)

    x = layers.add([x, input_tensor])
    return x

def conv_block_original_3l(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=2):
    filters1, filters2, filters3 = filters

    bn_axis = -1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv1D(filters1, 1, strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(filters3, 1,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv1D(filters3, 1, strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def conv_block_new_3l(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=2):
    filters1, filters2, filters3 = filters

    bn_axis = -1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + 'a')(input_tensor)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(filters1, 1, strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + 'a')(x)

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + 'b')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + 'b')(x)

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + 'c')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(filters3, 1, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + 'c')(x)

    shortcut = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + 'shortcut')(input_tensor)
    shortcut = layers.Activation('relu')(shortcut)
    shortcut = layers.Conv1D(filters3, 1, strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + 'shortcut')(shortcut)

    x = layers.add([x, shortcut])
    return x

class AMSoftmax(Layer):

    def __init__(self, output_dim, s, m, **kwargs):
        self.output_dim = output_dim
        self.s = s
        self.m = m
        super(AMSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][-1], self.output_dim),
                                      initializer=TruncatedNormal(mean=0.0, stddev=1.0),
                                      trainable=True)

        self.bias = self.add_weight(name='bias',
    	 			    shape=(self.output_dim, ),
				    initializer=TruncatedNormal(mean=0.0, stddev=1.0),
 		                    trainable=True)

        super(AMSoftmax, self).build(input_shape)

    def call(self, inputs):
        x = inputs[0]
        y = inputs[1]
        kernel_norm = K.l2_normalize(self.kernel, 0)
        cos_theta = K.dot(x, kernel_norm)
        cos_theta = K.clip(cos_theta, -1,1) # for numerical steady
        cos_theta = K.bias_add(cos_theta, self.bias, data_format='channels_last')
        phi = cos_theta - self.m
        adjust_theta = self.s * K.tf.where(K.tf.equal(y, 1), phi, cos_theta)
        return K.softmax(adjust_theta)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.output_dim)
#******************************************************************************#
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--src", required=True, help="source file")
ap.add_argument("-d", "--dst", required=True, help="destination dir")
ap.add_argument("-f", "--filename", required=True, help="output filename")
args = vars(ap.parse_args())
data_set_path = args["src"]
dest_path = args["dst"]
filename = args["filename"]

history = []
lr=1e-3
epochs=100

hf = h5py.File(data_set_path, 'r')
X_train = np.expand_dims(hf.get('X_train'), axis=-1)
X_test = np.expand_dims(hf.get('X_test'), axis=-1)
y_train = np.eye(30)[hf.get('y_train')]
y_test = np.eye(30)[hf.get('y_test')]
hf.close()

inputs = layers.Input(shape=(X_train.shape[-2], 1), name='input')

x = conv_block_original_3l(inputs, 5, [8, 8, 16], stage=1, block='a', strides=2)
x = identity_block_original_3l(x, 5, [8, 8, 16], stage=1, block='b')

x = layers.Flatten()(x)
x = layers.Dense(30, activation='softmax')(x)

model = Model(inputs=inputs, outputs=x)
model.summary()

for i in range(1):
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr), metrics=['acc'])
    history.append(model.fit(x=X_train, y=y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=2).history)
    sys.stdout.flush()

fileObject = open(os.path.join(dest_path, filename + ".pkl"), 'wb')
pickle.dump(history, fileObject)
fileObject.close()

model.save(os.path.join(dest_path, filename + ".h5"))
