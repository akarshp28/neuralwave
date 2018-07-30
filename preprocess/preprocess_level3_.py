import os
import sys
import shutil
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

def convert_to(data_paths, label, dest_path, class_name, min_, rng):
    """Converts a dataset to tfrecords."""
    filename = os.path.join(dest_path, class_name + '.tfrecords')
    if not os.path.exists(os.path.join(dest_path)):
        os.makedirs(os.path.join(dest_path))

    print('Writing', filename)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(len(data_paths)):
            data_raw = np.loadtxt(open(data_paths[index], "rb"), delimiter=",").astype(np.float32)
            for i in range(540):
                data_raw[:, i] = scalers[i].transform(np.expand_dims(data_raw[:, i], axis=0))
            data_raw = (data_raw-min_)/rng

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

comm = MPI.COMM_WORLD
size = comm.Get_size() # new: gives number of ranks in comm
rank = comm.Get_rank()

tmp = None
tmp_data = []
train_tmp = None
test_tmp = None

if (rank == 0):
    if not os.path.exists(os.path.join(dest_path, "train")):
        os.makedirs(os.path.join(dest_path, "train"))
    if not os.path.exists(os.path.join(dest_path, "test")):
        os.makedirs(os.path.join(dest_path, "test"))

    X, y, classes = read_samples(src_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    num_classes = len(classes)

    print("train size:", len(X_train), "test size:", len(X_test))
    print("Calculating scalers for training data")

    scalers = []
    for i in range(540):
        scalers.append(StandardScaler(with_std = False))

    tmp = X_train
    tmp = [tmp[i:i+(math.ceil(len(tmp)/size))] for i in range(0, len(tmp), math.ceil(len(tmp)/size))]

tmp = comm.scatter(tmp, root=0)
arrays = [read_array(addr) for addr in tmp]
tmp_data.extend(comm.gather(arrays, root=0))

if (rank == 0):
    tmp_data = np.array(tmp_data)
    min_ = np.min(arrays)
    max_ = np.max(arrays)

    for j in range(540):
        scalers[j].partial_fit(tmp_data[:, :, j])

    rng = max_ - min_

    train_tmp = []
    test_tmp = []
    for i in range(num_classes):
        train_tmp.append([X_train[np.where( y_train == i ), i, classes[i], min_, rng])
        test_tmp.append([X_test[np.where( y_test == i ), i, classes[i], min_, rng])

train_tmp = comm.scatter(train_tmp, root=0)
convert_to(train_tmp[0], train_tmp[1], os.path.join(dest_path, "train"), train_tmp[2])

test_tmp = comm.scatter(test_tmp, root=0)
convert_to(test_tmp[0], test_tmp[1], os.path.join(dest_path, "test"), test_tmp[2])
