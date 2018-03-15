import matplotlib
matplotlib.use('Agg')
import numpy as np
import pickle
from os.path import join
import os
import multiprocessing
import h5py
from joblib import Parallel, delayed
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 0.5
import matplotlib.pyplot as plt
from keras.layers import Input, Conv1D, UpSampling1D, MaxPooling1D
from keras.models import Model
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle
import pandas as pd
import scipy.misc
from joblib import Parallel, delayed
import multiprocessing
from scipy.misc import imresize
import time
import errno

names = ['abhishek', 'ahmad', 'akarsh', 'avatar', 'chaitanya', 'champ', 'harshith', 'ishan',
   'kalvik', 'manish', 'nishad', 'pavan', 'phani', 'prabhu', 'raghu', 'rahul', 'sanjay', 'shuang',
   'subramaniam', 'sushal', 'temesgen', 'vinay']
num_people = len(names)

#data filepath
srcdir = '/home/mlbots/neuralwave/dataset/data.h5'
#destination filepath
dstdir = './data/train'
#weights filepath
filepath="ca_weights_new-64-0.00.hdf5"

num_people = len(names)
win_size = 90000
num_cols = 6
batch_size = 4

def graph_plot(i, j, data_batch, label_batch):
    data = data_batch[i]
    plt.plot(data[:, 0], color="#1f77b4", alpha=0.6)
    plt.plot(data[:, 1], color="#ff7f0e", alpha=0.6)
    plt.plot(data[:, 2], color="#2ca02c", alpha=0.6)
    plt.plot(data[:, 3], color="#d62728", alpha=0.6)
    plt.plot(data[:, 4], color="#9467bd", alpha=0.6)
    plt.plot(data[:, 5], color="#8c564b", alpha=0.6)
    fig = plt.gca()
    fig.set_frame_on(False)
    fig.set_xticks([])
    fig.set_yticks([])
    plt.axis('off')
    name = names[label_batch[i]]
    DIR = join(join(dstdir, "plot"), name)

    if not os.path.exists(DIR):
        try:
            os.makedirs(DIR)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Directory not created.')
            else:
                raise
        
    fname = name + '_'+ str(j+i) + ".png"
    plt.savefig(join(DIR,fname), bbox_inches='tight', pad_inches=0.0)
    plt.clf()

def gadf(i, j, data_batch, label_batch):
    x = data_batch[i]
    x = np.dot(x, np.sqrt(np.ones(x.shape)-np.square(x)).transpose()) - np.dot(np.sqrt(np.ones(x.shape)-np.square(x)), x.transpose())
    name = names[label_batch[i]]
    DIR = join(join(dstdir, "gadf"), name)
    
    if not os.path.exists(DIR):
        try:
            os.makedirs(DIR)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Directory not created.')
            else:
                raise
            
    x = imresize(x, (500, 500))
    fname = name + '_'+ str(j+i) + ".png"
    plt.imsave(join(DIR,fname), x)

def gasf(i, j, data_batch, label_batch):
    x = data_batch[i]
    x = np.dot(x,x.transpose()) - np.dot(np.sqrt(np.ones(x.shape)-np.square(x)), np.sqrt(np.ones(x.shape)-np.square(x)).transpose())
    name = names[label_batch[i]]
    DIR = join(join(dstdir, "gasf"), name)
    
    if not os.path.exists(DIR):
        try:
            os.makedirs(DIR)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Directory not created.')
            else:
                raise
		    
    x = imresize(x, (500, 500))
    fname = name + '_'+ str(j+i) + ".png"
    plt.imsave(join(DIR,fname), x)

def scale_big_data(us_data, minn=None, maxx=None):
    int_a = -1
    int_b = 1
    sc_data = (((int_b - int_a)*(us_data-minn))/(maxx-minn)) + int_a
    return sc_data
		    
def main():
	print('importing dataset')

	with h5py.File(srcdir + '/data.h5', 'r') as hf:
        train_x = hf['train_x'][:]
        train_y = hf['train_y'][:]
        val_x = hf['val_x'][:]
        val_y = hf['val_y'][:]


	print(train_x.shape, train_y.shape, val_x.shape, val_y.shape)
	
if decision == 1:
    print('standardising and normalising dataset to [-1,1]')

    x_data_mag = train_x[:, 0:3]
    x_data_phase = train_x[:, 3:]

    x_data_mag -= np.mean(x_data_mag)
    x_data_phase -= np.mean(x_data_phase)

    # first 3 cols
    minel = np.min(x_data_mag)
    maxel = np.max(x_data_mag)
    x_data_mag = scale_big_data(x_data_mag, minn=minel, maxx=maxel)

    # last 3 cols
    minel = np.min(x_data_phase)
    maxel = np.max(x_data_phase)
    x_data_phase = scale_big_data(x_data_phase, minn=minel, maxx=maxel)

    train_x = np.concatenate((x_data_mag, x_data_phase), axis=1)
#############
    x_data_mag = val_x[:, 0:3]
    x_data_phase = val_x[:, 3:]

    x_data_mag -= np.mean(x_data_mag)
    x_data_phase -= np.mean(x_data_phase)

    # first 3 cols
    minel = np.min(x_data_mag)
    maxel = np.max(x_data_mag)
    x_data_mag = scale_big_data(x_data_mag, minn=minel, maxx=maxel)

    # last 3 cols
    minel = np.min(x_data_phase)
    maxel = np.max(x_data_phase)
    x_data_phase = scale_big_data(x_data_phase, minn=minel, maxx=maxel)

    val_x = np.concatenate((x_data_mag, x_data_phase), axis=1)

    print(train_x.shape, val_x.shape)

#############################################################################################

if decision == 2:
    print('standardising and normalising dataset to unit norm')

    x_data_mag = train_x[:, 0:3]
    x_data_phase = train_x[:, 3:]

    x_data_mag -= np.mean(x_data_mag)
    x_data_phase -= np.mean(x_data_phase)

    x_data_mag /= np.std(x_data_mag)
    x_data_phase /= np.std(x_data_phase)

    train_x = np.concatenate((x_data_mag, x_data_phase), axis=1)
########
    x_data_mag = val_x[:, 0:3]
    x_data_phase = val_x[:, 3:]

    x_data_mag -= np.mean(x_data_mag)
    x_data_phase -= np.mean(x_data_phase)

    x_data_mag /= np.std(x_data_mag)
    x_data_phase /= np.std(x_data_phase)

    val_x = np.concatenate((x_data_mag, x_data_phase), axis=1)

    print(train_x.shape, val_x.shape)

#############################################################################################

	total_batches = train_x.shape[0]
	
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

	model = Model(inputs, x)
	model.load_weights(filepath, by_name=True)
	model.summary()
	
	num_cores = multiprocessing.cpu_count()

	print("start")
	for i in range(0, train_x.shape[0], batch_size):
            print(train_x.shape[0], "/", i)

            x = train_x[i: i + batch_size]
            y = train_y[i: i + batch_size]

            Parallel(n_jobs=num_cores)(delayed(graph_plot)(j, i, x, y) for j in range(x.shape[0]))
            data_batch = model.predict_on_batch(x) 

            mean = np.mean(data_batch, axis = 1, keepdims=True)
            for j in range(data_batch.shape[0]):
                data_batch[j] -= mean[j] 

            minel = np.min(data_batch, axis = 1)
            maxel = np.max(data_batch, axis = 1)
            for j in range(data_batch.shape[0]):
                data_batch[j] = scale_big_data(data_batch[j], minn=minel[j], maxx=maxel[j])

            Parallel(n_jobs=num_cores)(delayed(gasf)(j, i, data_batch, y) for j in range(data_batch.shape[0]))
            Parallel(n_jobs=num_cores)(delayed(gadf)(j, i, data_batch, y) for j in range(data_batch.shape[0]))
			
if __name__ == "__main__": 
	main()

