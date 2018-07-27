import os
import sys
import shutil
import argparse
import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Instantiate the parser
parser = argparse.ArgumentParser()
# Optional positional argument
parser.add_argument('--src',
                    help='path of raw .dat files folder',
                    required=False,
                    default="/home/kalvik/shared/CSI_DATA/preprocessed_level2/")
# Optional argument
parser.add_argument('--dest',
                    help='path of destination folder',
                    required=False,
                    default="/home/kalvik/shared/CSI_DATA/tfrecords/")
# Optional argument
parser.add_argument('--jobs',
                    type=int,
                    help='number of jobs for parallization',
                    required=False,
                    default=16)

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

def convert_to(data_paths, label, dest_path, class_name):
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

def main(src_path, dest_path, n_jobs):
    X, y, classes = read_samples(src_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    num_classes = len(classes)

    print("train size:", len(X_train), "test size:", len(X_test))
    print("Calculating scalers for training data")
    scalers = []
    min_ = float('Inf')
    max_ = -float('Inf')
    for i in range(540):
        scalers.append(StandardScaler(with_std = False))

    for i in range(0, len(X_train), jobs):
        arrays = Parallel(n_jobs=jobs, verbose=0)(delayed(read_array)(addr) for addr in X_train[i:i+jobs-1])
        arrays = np.array(arrays)
        min_temp = np.min(arrays)
        max_temp = np.max(arrays)
        min_ = min(min_temp, min_)
        max_ = max(max_temp, max_)

        for j in range(540):
            scalers[j].partial_fit(arrays[:, :, j])

        sys.stdout.write("\r{}/{}".format(len(X_train), i+arrays.shape[0]))
        sys.stdout.flush()

    rng = max_ - min_
    print("range: ", rng)

    if not os.path.exists(os.path.join(dest_path, "train")):
        os.makedirs(os.path.join(dest_path, "train"))
    if not os.path.exists(os.path.join(dest_path, "test")):
        os.makedirs(os.path.join(dest_path, "test"))

    _ = Parallel(n_jobs=jobs, verbose=0)(delayed(convert_to)(X_train[np.where( y_train == i )], i,  os.path.join(dest_path, "train"), classes[i]) for i in range(num_classes))
    _ = Parallel(n_jobs=jobs, verbose=0)(delayed(convert_to)(X_test[np.where( y_test == i )], i,  os.path.join(dest_path, "test"), classes[i]) for i in range(num_classes))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.src, args.dest, args.jobs)
