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
lr = 0.0002
epochs = 200
G = 8

hf = h5py.File(data_set_path, 'r')
X_train = np.expand_dims(hf.get('X_train'), axis=-1)
X_test = np.expand_dims(hf.get('X_test'), axis=-1)
y_train = np.array(hf.get('y_train'))
y_test = np.array(hf.get('y_test'))
hf.close()

print(X_train.shape, X_test.shape)

class_trainx = []
class_trainy = []
class_testx = []
class_testy = []
for i in range(30):
	class_trainx.append(X_train[np.where(y_train == i)])
	class_trainy.append(y_train[np.where(y_train == i)])
	class_testx.append(X_test[np.where(y_test == i)])
	class_testy.append(y_test[np.where(y_test == i)])

final_train_x = []
final_train_y = []
final_test_x = []
final_test_y = []
for i in range(30):
	for _ in range(160):
		final_train_x.append(np.concatenate((class_trainx[i][np.random.randint(0, len(class_trainx[i]))], class_trainx[i][np.random.randint(0, len(class_trainx[i]))]), axis=-1))
		final_train_y.append(0)

	for _ in range(40):
		final_test_x.append(np.concatenate((class_testx[i][np.random.randint(0, len(class_testx[i]))], class_testx[i][np.random.randint(0, len(class_testx[i]))]), axis=-1))
		final_test_y.append(0)

for i in range(30):
	for _ in range(160):
		tmp1 = tmp2 = 1
		while (tmp1 == tmp2):
			tmp1 = np.random.randint(0, 30)
			tmp2 = np.random.randint(0, 30)
		final_train_x.append(np.concatenate((class_trainx[tmp1][np.random.randint(0, len(class_trainx[tmp1]))], class_trainx[tmp2][np.random.randint(0, len(class_trainx[tmp2]))]), axis=-1))
		final_train_y.append(1)

	for _ in range(40):
		tmp1 = tmp2 = 1
		while (tmp1 == tmp2):
			tmp1 = np.random.randint(0, 30)
			tmp2 = np.random.randint(0, 30)

		final_test_x.append(np.concatenate((class_testx[tmp1][np.random.randint(0, len(class_testx[tmp1]))], class_testx[tmp2][np.random.randint(0, len(class_testx[tmp2]))]), axis=-1))
		final_test_y.append(1)

final_train_x = np.array(final_train_x)
final_train_y = np.eye(2)[final_train_y]
final_test_x = np.array(final_test_x)
final_test_y = np.eye(2)[final_test_y]
print(final_train_x.shape, final_test_x.shape)

inputs = layers.Input(shape=(final_train_x.shape[1:]), name='input')

x = layers.ZeroPadding2D(padding=((0,0), (1,1)))(inputs)

x = layers.BatchNormalization(axis=-1, name='block1_bn1')(x)
x = layers.Activation('relu', name='block1__relu1')(x)
x = layers.Conv2D(64, (3, 3), padding='same', name='block1_conv1')(x)
x = layers.BatchNormalization(axis=-1, name='block1_bn2')(x)
x = layers.Activation('relu', name='block1__relu2')(x)
x = layers.Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

x = layers.BatchNormalization(axis=-1, name='block2_bn1')(x)
x = layers.Activation('relu', name='block2__relu1')(x)
x = layers.Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
x = layers.BatchNormalization(axis=-1, name='block2_bn2')(x)
x = layers.Activation('relu', name='block2__relu2')(x)
x = layers.Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

x = layers.BatchNormalization(axis=-1, name='block3_bn1')(x)
x = layers.Activation('relu', name='block3__relu1')(x)
x = layers.Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
x = layers.BatchNormalization(axis=-1, name='block3_bn2')(x)
x = layers.Activation('relu', name='block3__relu2')(x)
x = layers.Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
x = layers.BatchNormalization(axis=-1, name='block3_bn3')(x)
x = layers.Activation('relu', name='block3__relu3')(x)
x = layers.Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(2, activation='softmax', name='pred')(x)

base_model = Model(inputs=inputs, outputs=x)
base_model.summary()
model = multi_gpu_model(base_model, gpus=G)

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Nadam(lr=lr), metrics=['acc'])
model.fit(x=final_train_x, y=final_train_y, epochs=epochs, validation_data=(final_test_x, final_test_y), verbose=2, batch_size=128)

base_model.save('/users/kjakkala/neuralwave/data/vggsim.h5')
