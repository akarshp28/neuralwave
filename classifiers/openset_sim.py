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

def predict(x):
	preds = []
	for i in range(30):
		preds.append(model.predict(np.expand_dims(np.concatenate([x, dataset[i]], axis=-1), axis=0))[0])
	new_preds = []
	new_labels = []
	for i in range(len(preds)):
		if np.argmax(preds[i]) == 0:
			new_labels.append(i)
			new_preds.append(np.max(preds[i]))
	if (len(new_preds) > 0):
		label = new_labels[np.argmax(np.array(new_preds))]
		return label
	else:
		return -1

hf = h5py.File(data_set_path, 'r')
X_train = np.expand_dims(hf.get('X_train'), axis=-1)
X_test = np.expand_dims(hf.get('X_test'), axis=-1)
y_train = np.array(hf.get('y_train'))
y_test = np.array(hf.get('y_test'))
hf.close()

print(X_train.shape, X_test.shape)

model = load_model('/users/kjakkala/neuralwave/data/vggsim.h5')
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['acc'])

for ind in range(10):
	dataset = []
	for i in range(30):
		dataset.append(X_test[np.where(y_test == i)][ind])
	y_pred = []
	for i in range(X_test.shape[0]):
		y_pred.append(predict(X_test[i]))
	print(np.mean(np.equal(y_pred, y_test)))
