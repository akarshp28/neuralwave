from model import identity_block_1D, conv_block_1D
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import h5py
import matplotlib.pyplot as plt

data_dir="/home/kjakkala/neuralwave/data/CSI_preprocessed.h5"

hf = h5py.File(data_dir, 'r')
X_train = np.expand_dims(hf.get('X_train'), axis=-1)[:, 10:-10, 0]
X_test = np.expand_dims(hf.get('X_test'), axis=-1)[:, 10:-10, 0]
hf.close()

model = load_model("/home/kjakkala/neuralwave/classifiers/resnet.h5")

for i in range(10):
	x = model.predict(np.expand_dims(X_test[i], axis=0))
	plt.plot(x[0, :, 0])
	plt.plot(X_test[i, :, 0])
	plt.show()
