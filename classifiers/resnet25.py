from tensorflow import keras
from keras.initializers import TruncatedNormal
from keras.models import Model
from keras import optimizers
from keras import layers
import numpy as np
import pickle
import h5py
import math

def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
        
    bn_axis = -1
        
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv1D(filters1, 1,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(filters3, 1,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
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

#******************************************************************************#

hf = h5py.File('/scratch/kjakkala/neuralwave/data/pca_data.h5', 'r')
X_train = np.expand_dims(hf.get('X_train'), axis=-1)
X_test = np.expand_dims(hf.get('X_test'), axis=-1)
y_train = np.eye(30)[hf.get('y_train')]
y_test = np.eye(30)[hf.get('y_test')]
hf.close()

lr=1e-3
epochs=500

inputs = layers.Input(shape=(X_train.shape[-2], 1), name='input')

x = conv_block(inputs, 3, [4, 4, 8], stage=1, block='a', strides=2)
x = identity_block(x, 3, [4, 4, 8], stage=1, block='b')

x = conv_block(inputs, 3, [8, 8, 16], stage=2, block='a', strides=2)
x = identity_block(x, 3, [8, 8, 16], stage=2, block='b')

x = layers.Flatten()(x)
x = layers.Dense(30, activation='softmax')(x)

model = Model(inputs=inputs, outputs=x)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr, decay=1e-5), metrics=['acc'])
history = model.fit(x=X_train, y=y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=2)

#model.save("/users/kjakkala/resnet25_softmax.h5")

#fileObject = open("/users/kjakkala/resnet25_softmax.pkl", 'wb') 
#pickle.dump(history.history,fileObject)   
#fileObject.close()
