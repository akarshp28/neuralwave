#!/usr/bin/env python3
from tensorflow.python.client import device_lib
import tensorflow as tf
import numpy as np
import random
import time
import sys
import os

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])

#convert time in seconds to minutes or hours
def get_time(time):
    if time < 60:
        string = "{:.0f}s".format(time)
    elif time < 3600:
        string = "{:.0f}m".format(time/60)
    else:
        string = "{:.2f}h".format(time/60/60)
    return string

def identity_block(input_tensor, kernel_size, filters, stage, block, training):
    with tf.name_scope("stage-{}_block-{}".format(stage, block)):
        x = tf.layers.batch_normalization(input_tensor, training=training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, filters[0], kernel_size[0], padding="same")

        x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, filters[1], kernel_size[1], padding="same")

        x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, filters[2], kernel_size[2], padding="same")

        x = tf.add(x, input_tensor)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, training, strides=2):
    with tf.name_scope("stage-{}_block-{}".format(stage, block)):
        x = tf.layers.batch_normalization(input_tensor, training=training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, filters[0], kernel_size[0], strides=strides, padding="same")

        x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, filters[1], kernel_size[1], padding="same")

        x = tf.layers.batch_normalization(x, training=training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, filters[2], kernel_size[2], padding="same")

        shortcut = tf.layers.batch_normalization(input_tensor, training=training)
        shortcut = tf.nn.relu(shortcut)
        shortcut = tf.layers.conv2d(shortcut, filters[2], kernel_size[2], strides=strides, padding="same")

        x = tf.add(x, shortcut)
    return x

def classifier(i, inputs, labels, is_training):
    with tf.name_scope('resnet50_{}'.format(i)):
        with tf.name_scope("stage-1_block-a"):
            x = tf.layers.conv2d(inputs, 64, 7, strides=2, padding="same")
            x = tf.layers.batch_normalization(x, training=is_training)
            x = tf.nn.relu(x)
            x = tf.layers.max_pooling2d(x, 3, strides=2, padding="same")

        x = conv_block    (x, [1, 3, 1], [64, 64, 256], stage=2, block='a', training=is_training)
        x = identity_block(x, [1, 3, 1], [64, 64, 256], stage=2, block='b', training=is_training)
        x = identity_block(x, [1, 3, 1], [64, 64, 256], stage=2, block='c', training=is_training)

        x = conv_block    (x, [1, 3, 1], [128, 128, 512], stage=3, block='a', training=is_training)
        x = identity_block(x, [1, 3, 1], [128, 128, 512], stage=3, block='b', training=is_training)
        x = identity_block(x, [1, 3, 1], [128, 128, 512], stage=3, block='c', training=is_training)
        x = identity_block(x, [1, 3, 1], [128, 128, 512], stage=3, block='d', training=is_training)

        x = conv_block    (x, [1, 3, 1], [256, 256, 1024], stage=4, block='a', training=is_training)
        x = identity_block(x, [1, 3, 1], [256, 256, 1024], stage=4, block='b', training=is_training)
        x = identity_block(x, [1, 3, 1], [256, 256, 1024], stage=4, block='c', training=is_training)
        x = identity_block(x, [1, 3, 1], [256, 256, 1024], stage=4, block='d', training=is_training)
        x = identity_block(x, [1, 3, 1], [256, 256, 1024], stage=4, block='e', training=is_training)
        x = identity_block(x, [1, 3, 1], [256, 256, 1024], stage=4, block='f', training=is_training)

        x = conv_block    (x, [1, 3, 1], [512, 512, 2048], stage=5, block='a', training=is_training)
        x = identity_block(x, [1, 3, 1], [512, 512, 2048], stage=5, block='b', training=is_training)
        x = identity_block(x, [1, 3, 1], [512, 512, 2048], stage=5, block='c', training=is_training)

        x = tf.layers.average_pooling2d(x, (x.get_shape()[-3], x.get_shape()[-2]), 1) #global average pooling
        x = tf.layers.flatten(x)
        logits = tf.layers.dense(x, num_classes, activation=None)
        logits = tf.clip_by_value(logits, 1e-7, 1)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
        loss = tf.reduce_mean(cross_entropy)

        predicted = tf.nn.softmax(logits)
        correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return loss, accuracy

#function to read data in tfrecord files, convert to approiate format and reshape it
def _parse_function(example_proto):
    features = {"data": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.int64 , default_value=0)}
    parsed_features = tf.parse_single_example(example_proto, features)

    data = tf.decode_raw(parsed_features['data'], tf.float32)
    data = tf.reshape(data, [8000, 540])
    data = tf.expand_dims(data, axis=-1)
    data = data[3000:5000]

    label = tf.cast(parsed_features['label'], tf.int32)
    label = tf.one_hot(label, num_classes)
    return data, label

#Trainable autoencoder model with multi gpu training support
def model():
    #tf dataset to read tfrecord files
    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=num_classes)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat(-1)
    dataset = dataset.shuffle(buffer_size=(5 * batch_size * num_gpus))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1 * batch_size * num_gpus)
    iterator = dataset.make_initializable_iterator()

    is_training = tf.placeholder(tf.bool)

    #load model onto each gpu
    losses = []
    accuracies = []
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        for i in range(num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('Tower_%d' % (i)) as scope:
                    inputs, labels = iterator.get_next()
                    loss, acc = classifier(i, inputs, labels, is_training)
                    losses.append(loss)
                    accuracies.append(acc)

    avg_loss = tf.reduce_mean(losses)
    tf.summary.scalar("loss", avg_loss)

    avg_acc = tf.reduce_mean(accuracies)
    tf.summary.scalar("acc", avg_acc)

    ## Optimizer
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(lr,
                                               global_step,
                                               train_steps,
                                               decay_rate,
                                               staircase=True)
    tf.summary.scalar("learning_rate", learning_rate)

    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    apply_gradient_op = opt.minimize(avg_loss, global_step)

    #model saver, tensorboard, model initializer
    saver = tf.train.Saver(tf.global_variables())
    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    return init, merged, saver, global_step, avg_loss, avg_acc, apply_gradient_op, labels, iterator, filenames, is_training

train_path = "/scratch/kjakkala/preprocess_level3/train/"
test_path = "/scratch/kjakkala/preprocess_level3/test/"
weight_path = "/scratch/kjakkala/weights/resnet50/"
tensorboard_path = "/scratch/kjakkala/tensorboard/resnet50_"
sequence_length = 2000
input_width = 540
decay_rate = 0.96
batch_size = 1
save_epoch = 2
epochs = 50
lr = 5e-4
train_samples = 1096
test_samples = 194
num_gpus = get_available_gpus()
train_filenames = [train_path+file for file in os.listdir(train_path)]
test_filenames = [test_path+file for file in os.listdir(test_path)]
num_classes = len(train_filenames)
train_steps = int(train_samples//(batch_size*num_gpus))
test_steps = int(test_samples//(batch_size*num_gpus))

tf.reset_default_graph()
with tf.Graph().as_default(), tf.device('/cpu:0'):
    init, merged, saver, global_step, avg_loss, avg_acc, apply_gradient_op, labels, iterator, filenames, is_training = model()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if tf.train.latest_checkpoint(weight_path) != None:
            saver.restore(sess, tf.train.latest_checkpoint(weight_path))
        else:
            sess.run(init)

        train_writer = tf.summary.FileWriter(tensorboard_path+"train", sess.graph)
        test_writer = tf.summary.FileWriter(tensorboard_path+"test", sess.graph)

        for epoch in range(1, epochs + 1):
            print("\n\nEpoch {}/{}".format(epoch, epochs))
            batch_time = []
            epoch_time = time.time()

            #Training
            print("Training:")
            sess.run(iterator.initializer, feed_dict={filenames: train_filenames})
            for step in range(1, train_steps + 1):
                time_start = time.time()
                _, batch_loss, batch_acc, summary, g_step = sess.run([apply_gradient_op, avg_loss, avg_acc, merged, global_step], feed_dict={is_training: True})
                batch_time.append(time.time()-time_start)

                train_writer.add_summary(summary, g_step)

                if (step == train_steps):
                    sys.stdout.write("\r - {}/{} - {} - loss: {:.4f} - accuracy: {:.4f}".format(step, train_steps, get_time(time.time()-epoch_time), batch_loss, batch_acc))
                else:
                    sys.stdout.write("\r - {}/{} - {} - loss: {:.4f} - accuracy: {:.4f}".format(step, train_steps, get_time(np.mean(batch_time)*(train_steps-step)), batch_loss, batch_acc))
                sys.stdout.flush()

            #Testing
            batch_time = []
            epoch_time = time.time()
            print("\nTesting:")
            sess.run(iterator.initializer, feed_dict={filenames: test_filenames})
            for step in range(1, test_steps + 1):
                time_start = time.time()
                batch_loss, batch_acc, summary = sess.run([avg_loss, avg_acc, merged], feed_dict={is_training: False})
                batch_time.append(time.time()-time_start)

                test_writer.add_summary(summary, ((epoch-1)*test_steps)+step)

                if (step == test_steps):
                    sys.stdout.write("\r - {}/{} - {} - loss: {:.4f} - accuracy: {:.4f}".format(step, test_steps, get_time(time.time()-epoch_time), batch_loss, batch_acc))
                else:
                    sys.stdout.write("\r - {}/{} - {} - loss: {:.4f} - accuracy: {:.4f}".format(step, test_steps, get_time(np.mean(batch_time)*(test_steps-step)), batch_loss, batch_acc))
                sys.stdout.flush()

            #save model
            if epoch % save_epoch == 0:
                saver.save(sess, os.path.join(weight_path, "resnet50_loss-{}_acc-{}_epoch-{}".format(batch_loss, batch_acc, epoch)), global_step=g_step)

        #save model
        saver.save(sess, os.path.join(weight_path, "resnet50_loss-{}_acc-{}_epoch-{}".format(batch_loss, batch_acc, epoch)), global_step=g_step)

        print("\nFinished!")
