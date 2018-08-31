import os
import h5py
import numpy as np
from sklearn.decomposition import PCA

def read_array(data_path):
    return np.loadtxt(open(data_path, "rb"), delimiter=",", dtype=np.float32)

def read_samples(dataset_path, endswith=".csv"):
    datapaths, labels = list(), list()
    label = 0
    classes = sorted(os.listdir(dataset_path))
    # List each sub-directory (the classes)
    for c in classes:
        c_dir = os.path.join(dataset_path, c)
        walk = os.listdir(c_dir)
        # Add each image to the training set
        for sample in walk:
            # Only keeps csv samples
            if sample.endswith(endswith):
                datapaths.append(os.path.join(c_dir, sample))
                labels.append(label)
        label += 1
    return np.array(datapaths), np.array(labels)

def main():
    src_path = "/scratch/kjakkala/neuralwave/data/preprocess_level2"
    dest_file = "/scratch/kjakkala/neuralwave/data/pca_data_new.h5"

    train_x = []
    train_y = []
    test_x = []
    test_y = []

    pca = PCA(n_components=0.95)

    X, y = read_samples(os.path.join(src_path, "train"))
    for i in range(len(X)):
        train_x.append(read_array(X[i]))
        train_y.append(y[i])
    train_x = pca.fit_transform(np.reshape(train_x, (len(X), -1)))

    X, y = read_samples(os.path.join(src_path, "test"))
    for i in range(len(X)):
        test_x.append(read_array(X[i]))
        test_y.append(y[i])
    test_x = pca.transform(np.reshape(test_x, (len(X), -1)))

    hf = h5py.File(dest_file, 'w')
    hf.create_dataset('X_train', data=train_x)
    hf.create_dataset('y_train', data=train_y)
    hf.create_dataset('X_test', data=test_x)
    hf.create_dataset('y_test', data=test_y)
    hf.close()

main()
