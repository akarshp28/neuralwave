from tensorflow import keras
from keras.initializers import TruncatedNormal
from keras.utils import multi_gpu_model
from keras.engine.topology import Layer
from keras.models import load_model
from keras.layers import Lambda
from keras import backend as K
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
lr = 0.001
epochs = 200
G = 8

hf = h5py.File(data_set_path, 'r')
X_train = np.expand_dims(hf.get('X_train'), axis=-1)
X_test = np.expand_dims(hf.get('X_test'), axis=-1)
y_train = np.array(hf.get('y_train'))
y_test = np.array(hf.get('y_test'))
hf.close()

print(X_train.shape, X_test.shape)

'''
model = load_model('/users/kjakkala/neuralwave/data/vgg16_weights.h5')
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr), metrics=['acc'])
print(model.evaluate(x=X_test, y=np.eye(30)[y_test]))
'''
models = []
for i in range(0, 30, 2):
	X_train_tmp = X_train[np.hstack([np.where(y_train == i), np.where(y_train == i+1)]).reshape(-1)]
	y_train_tmp = np.eye(2)[np.hstack([np.zeros(np.where(y_train == i)[0].shape, dtype=int), np.ones(np.where(y_train == i+1)[0].shape, dtype=int)])]

	X_test_tmp = X_test[np.hstack([np.where(y_test == i), np.where(y_test == i+1)]).reshape(-1)]
	y_test_tmp = np.eye(2)[np.hstack([np.zeros(np.where(y_test == i)[0].shape, dtype=int), np.ones(np.where(y_test == i+1)[0].shape, dtype=int)])]

	model = load_model('/users/kjakkala/neuralwave/data/vgg16_weights.h5')
	model.load_weights('/users/kjakkala/neuralwave/data/vgg16_weights.h5')
	x = layers.Dense(2, activation='softmax', name='pred')(model.layers[-2].output)
	models.append(Model(inputs=model.inputs, outputs=x))

	for layer in models[-1].layers[:-2]:
		layer.trainable = False

	models[-1].compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr), metrics=['acc'])
	models[-1].fit(x=X_train_tmp, y=y_train_tmp, epochs=epochs, validation_data=(X_test_tmp, y_test_tmp), verbose=0)

	print(models[-1].evaluate(x=X_test_tmp, y=y_test_tmp))

for i in range(len(models)):
	models[i].save('/users/kjakkala/neuralwave/data/openset/vgg16_openset_{}.h5'.format(i))
