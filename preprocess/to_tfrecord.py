import os
import argparse
import numpy as np
import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(src_path, dst_path, label):
    """Converts a dataset to tfrecords."""
    classes = sorted(os.listdir(src_path))
    src_path = os.path.join(src_path, classes[int(label)])

    _, class_name = os.path.split(src_path)
    filename = os.path.join(dst_path, class_name + '.tfrecords')

    src_files = os.listdir(src_path)
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    with tf.python_io.TFRecordWriter(filename) as writer:
        for file in (src_files):
            data = read_array(os.path.join(src_path, file))
            example = tf.train.Example(
              features=tf.train.Features(
                  feature={
                      'label': _int64_feature(int(label)),
                      'data': _bytes_feature(data.tostring())
                  }))
            writer.write(example.SerializeToString())

def read_array(data_path):
    return np.loadtxt(open(data_path, "rb"), delimiter=",", dtype=np.float32)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--src", required=True, help="source dir")
    ap.add_argument("-d", "--dst", required=True, help="destination dir")
    ap.add_argument("-l", "--label", required=True, help="data label")
    args = vars(ap.parse_args())

    convert_to(args["src"], args["dst"], args["label"])
