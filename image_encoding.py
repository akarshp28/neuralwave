import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['lines.linewidth'] = 0.5
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.misc import imresize
from os.path import join
import multiprocessing
import numpy as np
import pickle
import os
import h5py
import scipy.misc
import time
import errno

names = []
num_people = len(names)

#data filepath
srcdir = '/home/mlbots/neuralwave/dataset/data.h5'
#destination filepath
dstdir = './data/train'

num_people = len(names)
win_size = 90000
num_cols = 6
batch_size = 4

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

def main():
	print('importing dataset')

	with h5py.File(srcdir , 'r') as hf:
        train_x = hf['train_x'][:]
        train_y = hf['train_y'][:]

	print(train_x.shape, train_y.shape)
	
	num_cores = multiprocessing.cpu_count()

	print("start")
	for i in range(0, train_x.shape[0], batch_size):
            print(train_x.shape[0], "/", i)

            x = train_x[i: i + batch_size]
            y = train_y[i: i + batch_size]

            Parallel(n_jobs=num_cores)(delayed(gasf)(j, i, x, y) for j in range(x.shape[0]))
            Parallel(n_jobs=num_cores)(delayed(gadf)(j, i, x, y) for j in range(x.shape[0]))
			
if __name__ == "__main__": 
	main()
