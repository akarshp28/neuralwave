from tensorflow import keras
from keras.engine.topology import Layer
from keras.models import load_model
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

data_set_path = "/users/kjakkala/neuralwave/data/CSI_30_l2_AMP_500_NO_PCA.h5"

hf = h5py.File(data_set_path, 'r')
X_train = np.expand_dims(hf.get('X_train'), axis=-1)
X_test = np.expand_dims(hf.get('X_test'), axis=-1)
y_train = np.array(hf.get('y_train'))
y_test = np.array(hf.get('y_test'))
hf.close()

print(X_train.shape, X_test.shape)

models = []
for file in os.listdir('/users/kjakkala/neuralwave/data/openset'):
	if os.path.isfile(os.path.join('/users/kjakkala/neuralwave/data/openset', file)):
		models.append(load_model(os.path.join('/users/kjakkala/neuralwave/data/openset', file)))

def predict(x):
	if (x.ndim == 3):
		x = np.expand_dims(x, axis=0)
	preds = []
	for i in range(len(models)):
		preds.append(models[i].predict(x))
		print(preds[-1])

predict(X_test[0])
