from tensorflow import keras
from keras.models import Model
from keras import optimizers
from keras import layers
import numpy as np
import pickle
import h5py

def identity_block(input_tensor, kernel_size, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.BatchNormalization(axis=-1, name=bn_name_base + 'a')(input_tensor)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(filters, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + 'a')(x)

    x = layers.BatchNormalization(axis=-1, name=bn_name_base + 'b')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(filters, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + 'b')(x)

    x = layers.add([x, input_tensor])
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.BatchNormalization(axis=-1, name=bn_name_base + 'a')(input_tensor)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(filters, kernel_size, strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + 'a')(x)

    x = layers.BatchNormalization(axis=-1, name=bn_name_base + 'b')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(filters, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + 'b')(x)

    shortcut = layers.BatchNormalization(axis=-1, name=bn_name_base + 'shortcut')(input_tensor)
    shortcut = layers.Activation('relu')(shortcut)
    shortcut = layers.Conv1D(filters, kernel_size, strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + 'shortcut')(shortcut)

    x = layers.add([x, shortcut])
    return x

#******************************************************************************#

hf = h5py.File('/scratch/kjakkala/neuralwave/data/pca_data.h5', 'r')
X_train = np.expand_dims(hf.get('X_train'), axis=-1)
X_test = np.expand_dims(hf.get('X_test'), axis=-1)
y_train = np.eye(30)[hf.get('y_train')]
y_test = np.eye(30)[hf.get('y_test')]
hf.close()

lr=1e-4
epochs=500
enc_dim=128

inputs = layers.Input(shape=(X_train.shape[-2], 1), name='input')

x = conv_block(inputs, 11, 16, stage=1, block='a', strides=2)
x = identity_block(x, 11, 16, stage=1, block='b')

x = conv_block(inputs, 5, 16, stage=2, block='a', strides=2)
x = identity_block(x, 5, 16, stage=2, block='b')

x = layers.Flatten()(x)
x = layers.Dense(enc_dim, activation='relu')(x)
x = layers.Dense(30, activation='softmax')(x)

model = Model(inputs=inputs, outputs=x)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr, decay=1e-5), metrics=['acc'])
history = model.fit(x=X_train, y=y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=2)

model.save("/users/kjakkala/resnet25_softmax_pre.h5")

fileObject = open("/users/kjakkala/resnet25_softmax_pre.pkl", 'wb')
pickle.dump(history.history,fileObject)
fileObject.close()
