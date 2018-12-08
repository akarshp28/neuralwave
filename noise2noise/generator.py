from keras.utils import Sequence
import numpy as np
import random
import h5py

class NoisyImageGenerator(Sequence):
    def __init__(self, data_set_path, source_noise_model, target_noise_model, batch_size=32):
        self.source_noise_model = source_noise_model
        self.target_noise_model = target_noise_model
        self.batch_size = batch_size

        hf = h5py.File(data_set_path, 'r')
        self.data = np.expand_dims(hf.get('X_train'), axis=-1)
        hf.close()

        self.num = len(self.data.shape[0])
        self.data_size = self.data.shape[1:3]

    def __len__(self):
        return self.num // self.batch_size

    def __getitem__(self):
        batch_size = self.batch_size
        data_size = self.data_size
        x = np.zeros((batch_size, data_size[0], data_size[1], 1), dtype=np.float64)
        y = np.zeros((batch_size, data_size[0], data_size[1], 1), dtype=np.float64)
        sample_id = 0

        while True:
            data_index = random.randint(0, self.num)
            sample = self.data[data_index]
            h, w, _ = sample.shape

            if h >= data_size[0] and w >= data_size[1]:
                x[sample_id] = self.source_noise_model(sample)
                y[sample_id] = self.target_noise_model(sample)

                sample_id += 1

                if sample_id == batch_size:
                    return x, y

class ValGenerator(Sequence):
    def __init__(self, data_set_path, source_noise_model, target_noise_model):

        hf = h5py.File(data_set_path, 'r')
        self.data = np.expand_dims(hf.get('X_test'), axis=-1)
        hf.close()

        self.num = len(self.data.shape[0])
        self.data_size = self.data.shape[1:3]
        
        x = np.zeros((num, data_size[0], data_size[1], 1), dtype=np.float64)
        y = np.zeros((num, data_size[0], data_size[1], 1), dtype=np.float64)

        for idx in range(num):
            sample = self.data[idx]
            x[idx] = self.source_noise_model(sample)
            y[idx] = self.target_noise_model(sample)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    pass