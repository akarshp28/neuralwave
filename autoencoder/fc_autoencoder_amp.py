from tensorflow import keras
from keras.utils import multi_gpu_model
from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers, layers
from random import shuffle
import numpy as np
import os

rows = 8000
cols = 272
epochs_ = 30
batch_size = 32
train_path = "/scratch/kjakkala/neuralwave/data/preprocess_level2/train/"
test_path = "/scratch/kjakkala/neuralwave/data/preprocess_level2/test/"
gpus_ = 8

def read_samples(dataset_path, endswith=".csv"):
    datapaths, labels = list(), list()
    label = 0
    classes = sorted(os.listdir(dataset_path))
    # List each sub-directory (the classes)
    for c in classes:
        c_dir = os.path.join(dataset_path, c)
        walk = os.listdir(c_dir)
        # Add each image to the training set
        for sample in walk:
            # Only keeps csv samples
            if sample.endswith(endswith):
                datapaths.append(os.path.join(c_dir, sample))
                labels.append(label)
        label += 1
    return datapaths, labels

def generator(datapath, batch_size):
    datapaths = read_samples(datapath)[0]
    while True:
        shuffle(datapaths)
        for i in range(0, len(datapaths), batch_size):
            if (i+batch_size >= len(datapaths)):
                continue
            else:
                data = []
                for j in range(batch_size):
                    data.append(np.loadtxt(open(datapaths[i+j], "rb"), delimiter=",", dtype=np.float32)[:, :cols])
                data = np.expand_dims(data, axis=-1)
                yield np.array(data), np.array(data)

def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    bn_axis = -1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
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
               strides=(2, 2)):
    filters1, filters2, filters3 = filters
    bn_axis = -1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def upconv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    filters1, filters2, filters3 = filters
    bn_axis = -1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2DTranspose(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2DTranspose(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

input = Input(shape=(rows, cols, 1))
x = layers.Conv2D(64, (7, 7),
                  strides=(2, 2),
                  padding='same',
                  kernel_initializer='he_normal',
                  name='conv1')(input)
x = layers.BatchNormalization(axis=-1, name='bn_conv1')(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

x = conv_block(x, 3, [8, 8, 16], stage=2, block='a')
x = identity_block(x, 3, [8, 8, 16], stage=2, block='b')

x = conv_block(x, 3, [1, 16, 1], stage=3, block='a')
encoded = identity_block(x, 3, [1, 16, 1], stage=3, block='b')

x = upconv_block(encoded, 3, [1, 16, 1], stage=4, block='a')
x = identity_block(x, 3, [1, 16, 1], stage=4, block='b')

x = upconv_block(x, 3, [8, 8, 16], stage=5, block='a')
x = identity_block(x, 3, [8, 8, 16], stage=5, block='b')

x = layers.Conv2DTranspose(   64, (7, 7),
                              strides=(2, 2),
                              padding='same',
                              kernel_initializer='he_normal',
                              name='conv')(x)
x = layers.BatchNormalization(axis=-1, name='bn_conv')(x)
x = layers.Activation('relu')(x)

x = layers.Conv2DTranspose(1, (1, 1),
                  strides=(2, 2),
                  padding='same',
                  kernel_initializer='he_normal',
                  name='conv_last')(x)
decoded = layers.Activation('sigmoid')(x)

model = Model(inputs=input, outputs=decoded)
model = multi_gpu_model(model, gpus=gpus_)
model.compile(optimizer=optimizers.Adam(lr=1e-4, decay=1e-5), loss='mse')

model.fit_generator(  generator(train_path, batch_size),
                steps_per_epoch=int(1107//batch_size),
                epochs=epochs_,
                verbose=1,
                validation_data=generator(test_path, batch_size),
                validation_steps=int(196//batch_size))
model.save_weights('/scratch/kjakkala/neuralwave/data/fc_autoencoder_weights.h5')

