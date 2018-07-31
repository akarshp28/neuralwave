import os
import sys
import math
import numpy as np
from mpi4py import MPI
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
            data_raw = (data_raw - min_)/(max_ - min_)

            example = tf.train.Example(
              features=tf.train.Features(
                  feature={
                      'label': _int64_feature(int(label)),
                      'data': _bytes_feature(data_raw.tostring())
                  }))
            writer.write(example.SerializeToString())

            sys.stdout.write("\r{}/{}".format(len(data_paths), index+1))
            sys.stdout.flush()
    print("\n")

def read_array(data_path):
    return np.loadtxt(open(data_path, "rb"), delimiter=",")

def scale_data(data_path):
    array = np.loadtxt(open(data_path, "rb"), delimiter=",")

    for i in range(540):
        array[:, i] = scalers[i].transform(np.expand_dims(array[:, i], axis=0))

    path, file = os.path.split(data_path)
    _, class_name = os.path.split(path)

    np.savetxt((os.path.join(os.path.join(dest_path, class_name), file)), array.astype(np.float32), delimiter=",")

#******************************************************************************#

src_path = "/users/kjakkala/neuralwave/data/preprocess_level2"
dest_path = "/users/kjakkala/neuralwave/data/preprocess_level3"
rows = 8000
cols = 540

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

train_sl = None
train_array = None
train_c = None
test_c = None
if (rank == 0):
    X, y, classes = read_samples(src_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    print("train size:", len(X_train), "test size:", len(X_test))

    train_sl = [X_train[i:i+(math.ceil(len(X_train)/size))] for i in range(0, len(X_train), math.ceil(len(X_train)/size))]
    train_array = np.empty((size, math.ceil(len(X_train)/size), rows, cols), dtype=np.float32)

#distribute data paths to processes
train_sl = comm.scatter(train_sl, root=0)
#read data with all processes
sl_data = np.array([read_array(addr) for addr in train_sl], dtype=np.float32)
#send data back to proc 1 from all other procs
comm.Gatherv(sl_data, train_array, root=0)

comm.Barrier() #wait for all procs to finish
if (rank == 0):
    train_array = np.reshape(train_array, (-1, rows, cols))[:len(X_train)]
    scalers = [StandardScaler(with_std = False) for _ in range(cols)]
    min = np.min(train_array)
    max = np.max(train_array)

    for i in range(cols):
        scalers[i].fit(train_array[:, :, i])

    if not os.path.exists(os.path.join(dest_path, "train")):
        os.makedirs(os.path.join(dest_path, "train"))
    if not os.path.exists(os.path.join(dest_path, "test")):
        os.makedirs(os.path.join(dest_path, "test"))

    train_c = [[X_train[np.where( y_train == i )], i, os.path.join(dest_path, "train"), classes[i], min, max, scalers] for i in range(len(classes))]
    test_c = [[X_test[np.where( y_test == i )], i,  os.path.join(dest_path, "test"), classes[i], min, max, scalers] for i in range(len(classes))]

comm.Barrier() #wait for all procs to finish

#distribute data paths to processes
train_c = comm.scatter(train_c, root=0)
convert_to(train_c[0], train_c[1], train_c[2], train_c[3], train_c[4], train_c[5], train_c[6])

test_c = comm.scatter(test_c, root=0)
convert_to(test_c[0], test_c[1], test_c[2], test_c[3], test_c[4], test_c[5], test_c[6])
