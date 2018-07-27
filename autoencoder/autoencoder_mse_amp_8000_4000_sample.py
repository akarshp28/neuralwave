#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import random
import h5py
import time
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#convert time in seconds to minutes and hours
def get_time(time):
    if time < 60:
        string = "{:.0f}s".format(time)
    elif time < 3600:
        string = "{:.0f}m".format(time/60)
    else:
        string = "{:.2f}h".format(time/60/60)

    return string

# see https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L101
def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):

        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = [g for g, _ in grad_and_vars]
        grad = tf.reduce_mean(grads, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

#Seq2Seq Autoencoder with a single LSTM for both encoder and decoder,
#along with optional rollout for the decoder LSTM
def autoencoder(i, inputs, rollout):

    #encoder lstm loop function, same as dynamic lstm in tf
    def encoder_loop_fn(time, cell_output, cell_state, loop_state):
        emit_output = cell_output  # == None for time == 0

        if cell_output is None:  # time == 0
            next_cell_state = encoder_cell.zero_state(batch_size, tf.float32)
        else:
            next_cell_state = cell_state

        elements_finished = (time >= sequence_length)
        finished = tf.reduce_all(elements_finished)

        next_input = tf.cond(finished,
                             lambda: tf.zeros([batch_size, input_width], dtype=tf.float32),
                             lambda: inputs_ta.read(time))

        next_loop_state = None
        return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

    #decoder lstm loop function, has rollout and fc for lstm outputs in order to
    #reshape outputs at the same shape as encoder inputs
    def decoder_loop_fn(time, cell_output, cell_state, loop_state):
        emit_output = cell_output

        if cell_output is None:  # time == 0
            next_cell_state = encoder_cell_states
            next_input = tf.zeros([batch_size, input_width], dtype=tf.float32)
        else:
            next_cell_state = cell_state
            next_input = tf.cond(rollout,
                                 lambda: tf.layers.dense(cell_output, input_width, activation=tf.nn.sigmoid, name="FC1", reuse=tf.AUTO_REUSE),
                                 lambda: inputs_ta.read(time-1))

        elements_finished = (time >= sequence_length)

        next_loop_state = None
        return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

    #autoencoder
    with tf.name_scope('autoencoder_{}'.format(i)):
        #convert sample into tensor array
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=sequence_length, clear_after_read=False)
        inputs_ta = inputs_ta.unstack(inputs)

        #encoder lstm
        with tf.name_scope('encoder'):
            encoder_cell = tf.contrib.rnn.LSTMCell(lstm_size)
            _, encoder_cell_states, _ = tf.nn.raw_rnn(encoder_cell, encoder_loop_fn)

        #decoder lstm
        with tf.name_scope('decoder'):
            decoder_cell = tf.contrib.rnn.LSTMCell(lstm_size)
            decoder_hidden_states_ta, _, _ = tf.nn.raw_rnn(decoder_cell, decoder_loop_fn)

        #convert lstm output array into a tensor
        outputs = decoder_hidden_states_ta.stack()
        outputs = tf.layers.dense(outputs, input_width, activation=tf.nn.sigmoid, name="FC1", reuse=tf.AUTO_REUSE)

    #mean squared error loss
    with tf.name_scope("loss_{}".format(i)):
        loss = tf.reduce_mean(tf.square(inputs-outputs))

    return encoder_cell_states, loss, outputs

#function to read data in tfrecord files, convert to approiate format and reshape it
def _parse_function(example_proto):
    features = {"data": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.int64 , default_value=0)}
    parsed_features = tf.parse_single_example(example_proto, features)
    data = tf.decode_raw(parsed_features['data'], tf.float32)
    data = tf.reshape(data, [8000, 540])

    data = data[:, :270]
    label = tf.cast(parsed_features['label'], tf.int32)
    return data, label

#Trainable autoencoder model with multi gpu training support
def model():
    #tf dataset to read tfrecord files
    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=num_classes)
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(4 * batch_size * num_gpus)
    iterator = dataset.make_initializable_iterator()

    #decoder lstm rollout state
    rollout = tf.placeholder(tf.bool)

    #load model onto each gpu
    losses = []
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        for i in range(num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('Tower_%d' % (i)) as scope:
                    # Dequeues one batch for the GPU
                    inputs, labels = iterator.get_next()
                    inputs = tf.transpose(inputs, perm=[2, 0, 1])
                    inputs.set_shape([sequence_length, batch_size, input_width])
                    encoder_cell_states, loss, outputs = autoencoder(i, inputs, rollout)
                    losses.append(loss)

    #average loss for tensorboard
    avg_loss = tf.reduce_mean(losses)

    #model saver, tensorboard, model initializer
    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()

    return init, saver, avg_loss, encoder_cell_states, labels, iterator, filenames, rollout, outputs

train_path = "/home/kalvik/shared/CSI_DATA/tfrecords/train/"
test_path = "/home/kalvik/shared/CSI_DATA/tfrecords/test/"
weight_path = "/home/kalvik/shared/neuralwave/autoencoder/weights/mse_amp_8000_4000/"
data_path = "/home/mse_amp_8000_4000_encoded.h5"

train_filenames = [train_path+file for file in os.listdir(train_path)]
test_filenames = [test_path+file for file in os.listdir(test_path)]
sequence_length = 270
input_width = 8000
lstm_size = 4000
batch_size = 8
num_gpus = 1
train_samples = 1096
test_samples = 194
num_classes = len(train_filenames)
train_steps = int(train_samples//(batch_size*num_gpus))
test_steps = int(test_samples//(batch_size*num_gpus))
batch_time = []
X, y = list(), list()

tf.reset_default_graph()
with tf.Graph().as_default(), tf.device('/cpu:0'):
    init, saver, avg_loss, encoder_cell_states, labels, iterator, filenames, rollout, outputs = model()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess, tf.train.latest_checkpoint(weight_path))

        print("started sampling training data:")
        sess.run(iterator.initializer, feed_dict={filenames: train_filenames})
        for step in range(1, train_steps + 1):
            time_start = time.time()
            x_temp, y_temp = sess.run([encoder_cell_states, labels], feed_dict={rollout:True})
            batch_time.append(time.time()-time_start)
            X.expand(x_temp)
            y.expand(y_temp)
            print("Batch Time: {}, Step: {}".format(get_time(mean(batch_time)), step))

        print("\nstarted sampling testing data:")
        sess.run(iterator.initializer, feed_dict={filenames: test_filenames})
        for step in range(1, test_steps + 1):
            time_start = time.time()
            x_temp, y_temp = sess.run([encoder_cell_states, labels], feed_dict={rollout:True})
            batch_time.append(time.time()-time_start)
            X.expand(x_temp)
            y.expand(y_temp)
            print("Batch Time: {}, Step: {}".format(get_time(mean(batch_time)), step))
        print("\nFinished!")

X = np.array(X)
y = np.array(y)
hf = h5py.File(data_path, 'w')
hf.create_dataset('X', data=X)
hf.create_dataset('y', data=y)
hf.close()
