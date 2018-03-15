from keras.layers import Input, Conv1D, UpSampling1D, MaxPooling1D
from keras.models import Model
import numpy as np
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from sklearn.utils import shuffle
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
learning_rate = 0.001

def scale_big_data(us_data, minn, maxx):
    int_a = -1
    int_b = 1
    sc_data = (((int_b - int_a)*(us_data-minn))/(maxx-minn)) + int_a
    return sc_data

#############################################################################################

print('importing dataset')

with h5py.File(srcdir + '/data.h5', 'r') as hf:
    train_x = hf['train_x'][:]
    train_y = hf['train_y'][:]
    val_x = hf['val_x'][:]
    val_y = hf['val_y'][:]


print(train_x.shape, train_y.shape, val_x.shape, val_y.shape)

#############################################################################################

print('standardising and normalising dataset to [-1,1]')
x_data_mag = train_x[:, :,:3]
x_data_phase = train_x[:, :,3:]

mean_mag = np.mean(np.mean(x_data_mag, axis=-2), axis=-2)
mean_phase = np.mean(np.mean(x_data_phase, axis=-2), axis=-2)

x_data_mag -= mean_mag
x_data_phase -= mean_phase

 # first 3 cols
minel_mag = np.min(np.min(x_data_mag, axis = -2), axis = -2)
maxel_mag = np.max(np.max(x_data_mag, axis = -2), axis = -2)
x_data_mag = scale_big_data(x_data_mag, minn=minel_mag, maxx=maxel_mag)

# last 3 cols
minel_phase = np.min(np.min(x_data_phase, axis = -2), axis = -2)
maxel_phase = np.max(np.max(x_data_phase, axis = -2), axis = -2)
x_data_phase = scale_big_data(x_data_phase, minn=minel_phase, maxx=maxel_phase)

train_x = np.concatenate((x_data_mag, x_data_phase), axis=-1)

# #############
x_data_mag = val_x[:, :,:3]
x_data_phase = val_x[:, :,3:]

x_data_mag -= mean_mag
x_data_phase -= mean_phase

# first 3 cols
x_data_mag = scale_big_data(x_data_mag, minn=minel_mag, maxx=maxel_mag)

# last 3 cols
x_data_phase = scale_big_data(x_data_phase, minn=minel_phase, maxx=maxel_phase)

val_x = np.concatenate((x_data_mag, x_data_phase), axis=-1)

print(train_x.shape, train_y.shape, val_x.shape, val_y.shape)

# #############################################################################################

def get_csi_minibatch(train_x, train_y, batch_size):
    while True:
        train_x, train_y = shuffle(train_x, train_y)
        for p in range(0, train_x.shape[0], batch_size):
            yy_data = train_x[p: p + batch_size]
            yy_label = train_y[p: p + batch_size]
            yield yy_data, yy_label

# ##########################################################################################

print('creating network')
inputs = Input(shape=(win_size, num_cols))

x = Conv1D(256, 200, padding="same", name="encoder_Conv1")(inputs)
x = MaxPooling1D(2, strides = 2, name= "encoder_max1")(x)

x = Conv1D(128, 200, padding="same", name="encoder_Conv2")(x)
x = MaxPooling1D(2, strides = 2, name= "encoder_max2")(x)

x = Conv1D(64, 200, padding="same", name="encoder_Conv3")(x)
x = MaxPooling1D(2, strides = 2, name= "encoder_max3")(x)

x = Conv1D(32, 200, padding="same", name="encoder_Conv4")(x)
encoder = MaxPooling1D(2, strides = 2, name= "encoder_max4")(x)


x = Conv1D(32, 200, padding="same", name="decoder_Conv1")(encoder)
x = UpSampling1D(size = 2, name="decoder_up1")(x)

x = Conv1D(64, 200, padding="same", name="decoder_Conv2")(x)
x = UpSampling1D(size = 2, name="decoder_up2")(x)

x = Conv1D(128, 200, padding="same", name="decoder_Conv3")(x)
x = UpSampling1D(size = 2, name="decoder_up3")(x)

x = Conv1D(256, 200, padding="same", name="decoder_Conv4")(x)
x = UpSampling1D(size = 2, name="decoder_up4")(x)

x = Conv1D(num_cols, 200, padding="same", name="reshape_conv")(x)

model = Model(inputs, x)
model.summary()

model.compile(optimizer=optimizers.adam(lr=learning_rate, decay=1e-5),
              loss='mean_squared_error',
              metrics=['accuracy'])

train = get_csi_minibatch(train_x, train_x, batch_size=batch_size)
validation = get_csi_minibatch(val_x, val_x, batch_size=batch_size)

tensorboard = TensorBoard(log_dir='./logs/autoencoder_weights_{0}'.format(time.time()),
                 histogram_freq=0,
                 write_graph=True,
                 write_images=False)

filepath="./weights/autoencoder_weights_{epoch:02d}_{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=5)

print('training started')

model.fit_generator(train,
                    steps_per_epoch = train_x.shape[0]//batch_size,
                    epochs = EPOCHS,
                    callbacks = [tensorboard, checkpoint],
                    validation_data = validation,
                    validation_steps = val_x.shape[0]//batch_size,
                    verbose=2)
