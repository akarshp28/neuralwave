from tensorflow import keras
from model import identity_block, conv_block
from keras.utils import multi_gpu_model
from keras.models import Model
from keras import optimizers
from keras import layers
import numpy as np
import argparse
import h5py

def get_args():
    parser = argparse.ArgumentParser(description="train resent on CSI data",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True,
                        help="h5py data file")
    parser.add_argument("--lr", type=float, required=True, default="1e-3",
                        help="initial learning rate")
    parser.add_argument("--decay", type=float, required=True, default="1e-2",
                        help="learning rate decay")
    parser.add_argument("--nb_epochs", type=int, required=True, default="25",
                        help="number of training epochs")
    args = parser.parse_args()

    return args

def main():
    args = get_args()
    lr=args.lr
    epochs=args.nb_epochs
    data_dir=args.data_dir
    decay=args.decay

    hf = h5py.File(data_dir, 'r')
    X_train = np.expand_dims(hf.get('X_train'), axis=-1)
    X_test = np.expand_dims(hf.get('X_test'), axis=-1)
    y_train = np.eye(40)[hf.get('y_train')]
    y_test = np.eye(40)[hf.get('y_test')]
    hf.close()

    inputs = layers.Input(shape=(None, None, 1))

    x = layers.Conv2D(64, (7, 7), strides=(2, 2))(inputs)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, [64, 256], "relu")
    x = identity_block(x, [64, 256], "relu")

    x = conv_block(x, [128, 512], "relu")
    x = identity_block(x, [128, 512], "relu")

    x = conv_block(x, [256, 1024], "relu")
    x = identity_block(x, [256, 1024], "relu")

    x = conv_block(x, [512, 2048], "relu")
    x = identity_block(x, [512, 2048], "relu")

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(40, activation='softmax')(x)

    model_base = Model(inputs=inputs, outputs=x)
    model_base.summary()
    model = multi_gpu_model(model_base, gpus=4)

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr, decay=decay), metrics=['acc'])
    history = model.fit(x=X_train, y=y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=2)

    model_base.save("resnet.h5")

if __name__ == '__main__':
    main()
