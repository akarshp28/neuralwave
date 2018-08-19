from tensorflow import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
from random import shuffle
import numpy as np
import os

encoding_dim = 4000
rows = 8000
cols = 270
epochs_ = 50
batch_size = 64
cpus = int(10/2)
train_path = "/scratch/kjakkala/neuralwave/data/preprocess_level2/train"
test_path = "/scratch/kjakkala/neuralwave/data/preprocess_level2/test"
G = keras.backend.tensorflow_backend._get_available_gpus()
print("GPUs:", G)

def read_samples(dataset_path, endswith=".csv"):
    datapaths, labels = list(), list()
    label = 0
    classes = sorted(os.walk(dataset_path).__next__()[1])

    # List each sub-directory (the classes)
    for c in classes:
        c_dir = os.path.join(dataset_path, c)
        walk = os.walk(c_dir).__next__()
        # Add each image to the training set
        for sample in walk[2]:
            # Only keeps csv samples
            if sample.endswith(endswith):
                datapaths.append(os.path.join(c_dir, sample))
                labels.append(label)
        label += 1

    return datapaths, labels

def generator(datapath, batch_size):
    datapaths = read_samples(datapath)
    while True:
        shuffle(datapaths)
        for i in range(0, len(datapaths), batch_size):
            for j in range(batch_size):
                data = []
                data.append(np.loadtxt(open(datapaths[i+j], "rb"), delimiter=",", dtype=np.float32))
                yield np.array(data), np.array(data)

input = Input(shape=(rows*cols,))
encoded = Dense(encoding_dim, activation='relu')(input)
decoded = Dense(rows*cols, activation='sigmoid')(encoded)

model = Model(inputs=input, outputs=decoded)
model = multi_gpu_model(model, gpus=G)
model.compile(optimizer=optimizers.Adam(lr=1e-4, decay=1e-5), loss='mse')

model.fit_generator(  generator(train_path, batch_size),
                steps_per_epoch=1107,
                epochs=epochs_,
                verbose=1,
                validation_data=generator(test_path, batch_size),
                validation_steps=196,
                workers=cpus,
                use_multiprocessing=True,
                shuffle=True)
model.save('/scratch/kjakkala/neuralwave/data/fc_autoencoder.h5')
