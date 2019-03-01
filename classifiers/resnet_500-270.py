from model import identity_block, conv_block
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras import layers
import numpy as np
import argparse
import pickle
import h5py

def get_args():
    parser = argparse.ArgumentParser(description="train resent on CSI data",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True, help="h5py data file")
    parser.add_argument("--save_path", type=str, required=True, help="h5py data file for weight")
    parser.add_argument("--history_path", type=str, required=True, help="pkl data file for history")
    parser.add_argument("--ngpu", type=int, required=True, help="number of gpus")
    args = parser.parse_args()

    return args

def main():
    args=get_args()
    data_dir=args.data_dir
    save_path=args.save_path
    history_path=args.history_path
    ngpu=args.ngpu

    lr=1e-4
    epochs=250
    decay=1e-2

    hf = h5py.File(data_dir, 'r')
    train_classes = np.array(hf.get('labels')).astype(str)
    num_classes = len(train_classes)
    X_train = np.expand_dims(hf.get('X_train'), axis=-1)
    X_test = np.expand_dims(hf.get('X_test'), axis=-1)
    y_train = np.eye(num_classes)[hf.get('y_train')]
    y_test = np.eye(num_classes)[hf.get('y_test')]
    hf.close()

    input_layer = layers.Input(shape=(X_train.shape[1:]))

    x = layers.Conv2D(64, (7, 7), strides=(2, 2))(input_layer)
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

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation=None)(x)
    x = layers.Activation('softmax')(x)

    model_base = Model(inputs=input_layer, outputs=x)
    model_base.summary()
    model = multi_gpu_model(model_base, gpus=ngpu)

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr, decay=decay), metrics=['acc'])
    history = model.fit(x=X_train, y=y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=2, batch_size=16)

    model_base.save(save_path)
    
    outfile = open(history_path,'wb')
    pickle.dump(history.history, outfile)
    outfile.close()

if __name__ == '__main__':
    main()

