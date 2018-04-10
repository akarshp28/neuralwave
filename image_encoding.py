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

num_people = 24

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

#############################################################################################
	
	num_cores = multiprocessing.cpu_count()

	print("start")
	for i in range(0, train_x.shape[0], batch_size):
            print(train_x.shape[0], "/", i)

            x = train_x[i: i + batch_size]
            y = train_y[i: i + batch_size]

            Parallel(n_jobs=num_cores)(delayed(graph_plot)(j, i, x, y) for j in range(x.shape[0]))

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
