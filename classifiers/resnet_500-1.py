from model import identity_block_1D, conv_block_1D
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras import layers
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
    X_train = np.expand_dims(hf.get('X_train'), axis=-1)[:, 10:-10, 0]
    X_test = np.expand_dims(hf.get('X_test'), axis=-1)[:, 10:-10, 0]
    hf.close()

    inputs = layers.Input(shape=(X_train.shape[1:]))

    x = layers.Conv1D(4, 3, strides=(2), padding='same')(inputs)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(3, strides=2, padding='same')(x)

    x = layers.Conv1D(4, 3, padding='same')(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(3, strides=2, padding='same')(x)

    x = layers.Conv1D(8, 3, padding='same')(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(3, strides=2, padding='same')(x)

    x = layers.Conv1D(8, 3, padding='same')(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(3, strides=2, padding='same')(x)

    x = layers.Conv1D(8, 3, padding='same')(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)

    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(8, 3, padding='same')(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)

    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(8, 3, padding='same')(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)

    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(8, 3, padding='same')(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)

    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(4, 3, padding='same')(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)

    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(4, 3, padding='same')(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(1, 3, padding='same')(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation('sigmoid')(x)

    model_base = Model(inputs=inputs, outputs=x)
    model_base.summary()
    model = multi_gpu_model(model_base, gpus=4)

    model.compile(loss='mse', optimizer=optimizers.Adam(lr=lr, decay=decay))
    history = model.fit(x=X_train, y=X_train, epochs=epochs, validation_data=(X_test, X_test), verbose=2)

    model_base.save("resnet.h5")

if __name__ == '__main__':
    main()
