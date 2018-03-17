from keras.layers import Input, Conv1D, UpSampling1D, MaxPooling1D
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Lambda
from sklearn.utils import shuffle
from keras.models import Model
from keras import backend as K
from keras import optimizers
import pandas as pd
import numpy as np
import time
import h5py

#############################################################################################################

names = ['abhishek', 'ahmad', 'akarsh', 'avatar', 'chaitanya', 'champ', 'harshith', 'ishan',
       'kalvik', 'manish', 'nishad', 'pavan', 'phani', 'prabhu', 'raghu', 'rahul', 'sanjay', 'shuang',
       'subramaniam', 'sushal', 'temesgen', 'vinay']

#############################################################################################################

srcdir = './dataset'

num_people = len(names)
print(num_people)
win_size = 90000
num_cols = 6
batch_size = 8
EPOCHS = 150
learning_rate = 0.01

def scale_big_data(us_data, minn, maxx):
    int_a = -1
    int_b = 1
    sc_data = (((int_b - int_a)*(us_data-minn))/(maxx-minn)) + int_a
    return sc_data

#############################################################################################

print('importing dataset')

with h5py.File(srcdir + '/data.h5', 'r') as hf:
    train_x = hf['train_x'][:]
    val_x = hf['val_x'][:]


print(train_x.shape, val_x.shape)

#############################################################################################

print('standardising and normalising dataset to [-1,1]')
x_data_mag = train_x[:, :, :3]
x_data_phase = train_x[:, :, 3:]

mean_mag = np.mean(x_data_mag)
mean_phase = np.mean(x_data_phase)

x_data_mag -= mean_mag
x_data_phase -= mean_phase

 # first 3 cols
minel_mag = np.min(x_data_mag)
maxel_mag = np.max(x_data_mag)
x_data_mag = scale_big_data(x_data_mag, minn=minel_mag, maxx=maxel_mag)

# last 3 cols
minel_phase = np.min(x_data_phase)
maxel_phase = np.max(x_data_phase)
x_data_phase = scale_big_data(x_data_phase, minn=minel_phase, maxx=maxel_phase)

train_x = np.concatenate((x_data_mag, x_data_phase), axis=-1)

# #############
x_data_mag = val_x[:, :, :3]
x_data_phase = val_x[:, :, 3:]

x_data_mag -= mean_mag
x_data_phase -= mean_phase

# first 3 cols
x_data_mag = scale_big_data(x_data_mag, minn=minel_mag, maxx=maxel_mag)

# last 3 cols
x_data_phase = scale_big_data(x_data_phase, minn=minel_phase, maxx=maxel_phase)

val_x = np.concatenate((x_data_mag, x_data_phase), axis=-1)

train_x = np.expand_dims(train_x, -1)
val_x = np.expand_dims(val_x, -1)

del x_data_mag, x_data_phase

print("Final preprocessed data shape: ", train_x.shape, val_x.shape)

# ##########################################################################################
def expand_dims(x):
    return K.expand_dims(x, -1)
    
def squeeze(x):
    return K.squeeze(x, -1)


print('creating network')
inputs = Input(shape=(win_size, num_cols, 1))

x = Lambda(squeeze)(inputs)

x = Conv1D(256, 200, padding="same", name="encoder_Conv1", activation="relu")(x)
x = MaxPooling1D(2, strides = 2, name= "encoder_max1")(x)

x = Conv1D(128, 200, padding="same", name="encoder_Conv2", activation="relu")(x)
x = MaxPooling1D(2, strides = 2, name= "encoder_max2")(x)

x = Conv1D(64, 200, padding="same", name="encoder_Conv3", activation="relu")(x)
x = MaxPooling1D(2, strides = 2, name= "encoder_max3")(x)

x = Conv1D(32, 200, padding="same", name="encoder_Conv4", activation="relu")(x)

encoder = MaxPooling1D(2, strides = 2, name= "encoder_max4")(x)


x = Conv1D(32, 200, padding="same", name="decoder_Conv1", activation="relu")(encoder)
x = UpSampling1D(size = 2, name="decoder_up1")(x)

x = Conv1D(64, 200, padding="same", name="decoder_Conv2", activation="relu")(x)
x = UpSampling1D(size = 2, name="decoder_up2")(x)

x = Conv1D(128, 200, padding="same", name="decoder_Conv3", activation="relu")(x)
x = UpSampling1D(size = 2, name="decoder_up3")(x)

x = Conv1D(256, 200, padding="same", name="decoder_Conv4", activation="relu")(x)
x = UpSampling1D(size = 2, name="decoder_up4")(x)

x = Conv1D(num_cols, 200, padding="same", name="reshape_conv", activation="tanh")(x)

x = Lambda(expand_dims)(x)

model = Model(inputs, x)
model.summary()

model.compile(optimizer=optimizers.Nadam(),
              loss='mean_squared_error',
              metrics=['accuracy'])

train_data = ImageDataGenerator()
val_data = ImageDataGenerator()

tensorboard = TensorBoard(log_dir='./logs/autoencoder_weights_{0}'.format(time.time()), write_graph=True)

filepath="./weights/autoencoder_weights_{epoch:02d}_{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=5)

print('training started')

model.fit_generator(train_data.flow(train_x, train_x, batch_size=batch_size),
                    steps_per_epoch = train_x.shape[0]//batch_size,
                    epochs = EPOCHS,
                    callbacks = [tensorboard, checkpoint],
                    validation_data = val_data.flow(val_x, val_x, batch_size=batch_size),
                    validation_steps = val_x.shape[0]//batch_size,
                    verbose = 1)
