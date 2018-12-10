import h5py
import numpy as np
from model import get_unet_model, PSNR
from tensorflow.keras.models import load_model

hf = h5py.File("/home/kjakkala/neuralwave/data/CSI_preprocessed.h5", 'r')
data = np.expand_dims(hf.get('X_test'), axis=-1)
hf.close()

model = load_model('/home/kjakkala/neuralwave/data/weights.015-0.044-13.63148.hdf5', custom_objects={'PSNR': PSNR})
pred = model.predict(np.expand_dims(np.pad(data[0], [(0,0),(1,1),(0,0)], 'constant', constant_values=0)[2:-2], axis=0))

hf = h5py.File('data.h5', 'w')
hf.create_dataset('pred', data=pred)
hf.create_dataset('orig', data=data[0])
hf.close()
