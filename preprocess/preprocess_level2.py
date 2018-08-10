import os
import sys
import math
import numpy as np
from mpi4py import MPI
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def smooth(x,window_len):
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    w=np.hanning(window_len)
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(window_len//2):-(window_len//2)]

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
    return np.array(datapaths), np.array(labels), classes

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(data_paths, label, dest_path, class_name, min_, max_, scalers):
    """Converts a dataset to tfrecords."""
    filename = os.path.join(dest_path, class_name + '.tfrecords')
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    print('Writing', filename)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(len(data_paths)):
            data_raw = np.loadtxt(open(data_paths[index], "rb"), delimiter=",").astype(np.float32)
            for i in range(540):
                data_raw[:, i] = scalers[i].transform(np.expand_dims(data_raw[:, i], axis=0))
                data_raw[:, i] = smooth(data_raw[:, i], 91)
            data_raw = (data_raw - min_)/(max_ - min_)

            example = tf.train.Example(
              features=tf.train.Features(
                  feature={
                      'label': _int64_feature(int(label)),
                      'data': _bytes_feature(data_raw.tostring())
                  }))
            writer.write(example.SerializeToString())

def read_array(data_path):
    return np.loadtxt(open(data_path, "rb"), delimiter=",", dtype=np.float32)

def scale_data(data_path):
    array = np.loadtxt(open(data_path, "rb"), delimiter=",")

    for i in range(540):
        array[:, i] = scalers[i].transform(np.expand_dims(array[:, i], axis=0))

    path, file = os.path.split(data_path)
    _, class_name = os.path.split(path)

    np.savetxt((os.path.join(os.path.join(dest_path, class_name), file)), array.astype(np.float32), delimiter=",")

#******************************************************************************#

src_path = "/users/kjakkala/neuralwave/data/preprocess_level1"
dest_path = "/users/kjakkala/neuralwave/data/preprocess_level2"
rows = 8000
cols = 540

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

X, y, classes = read_samples(src_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

num_sl = math.floor(len(X_train)/size)
train_sl = [None for _ in range(num_sl)]
train_array = None

data_c_len = None
data_c = [None for _ in range(len(classes)*2)]

if (rank == 0):
    print("train size:", len(X_train), "test size:", len(X_test))

    train_sl = [X_train[i:i+size] for i in range(0, len(X_train), size)]
    train_array = np.empty((size, rows, cols), dtype=np.float32)

    if (len(train_sl[-1]) != size):
        last_sl = train_sl[-1]
        del train_sl[-1]

    scalers = [StandardScaler(with_std = False) for _ in range(cols)]

    min_ = float('Inf')
    max_ = -float('Inf')

for index in range(num_sl):
    addr = comm.scatter(train_sl[index], root=0)
    comm.Gatherv(np.expand_dims(read_array(addr), axis=0), train_array, root=0)

    if (rank == 0):
        for i in range(cols):
            scalers[i].partial_fit(train_array[:, :, i])

        min_temp = np.min(train_array)
        max_temp = np.max(train_array)

        min_ = min(min_temp, min_)
        max_ = max(max_temp, max_)

if (rank == 0):
    train_array = np.array([read_array(addr) for addr in last_sl])

    for i in range(cols):
        scalers[i].partial_fit(train_array[:, :, i])

    min_temp = np.min(train_array)
    max_temp = np.max(train_array)

    min_ = min(min_temp, min_)
    max_ = max(max_temp, max_)

    if not os.path.exists(os.path.join(dest_path, "train")):
        os.makedirs(os.path.join(dest_path, "train"))
    if not os.path.exists(os.path.join(dest_path, "test")):
        os.makedirs(os.path.join(dest_path, "test"))

    data_tmp = [[X_train[np.where( y_train == i )], i, os.path.join(dest_path, "train"), classes[i], min_, max_, scalers] for i in range(len(classes))]
    data_tmp.extend([X_test[np.where( y_test == i )], i, os.path.join(dest_path, "test"), classes[i], min_, max_, scalers] for i in range(len(classes)))
    data_c = [data_tmp[i:i+size] for i in range(0, len(data_tmp), size)]

    if (len(data_c[-1]) < size):
        data_c_last = data_c[-1]
        del data_c[-1]

    data_c_len = len(data_c)

data_c_len = comm.bcast(data_c_len, root=0)

for i in range(data_c_len):
    data_tmp = comm.scatter(data_c[i], root=0)
    convert_to(data_tmp[0], data_tmp[1], data_tmp[2], data_tmp[3], data_tmp[4], data_tmp[5], data_tmp[6])

if (rank == 0):
    if (len(data_c_last) >= 1):
        for tmp in data_c_last:
            convert_to(data_tmp[0], data_tmp[1], data_tmp[2], data_tmp[3], data_tmp[4], data_tmp[5], data_tmp[6])
