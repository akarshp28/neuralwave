from tensorflow import keras
from keras.layers import Input, Conv1D, Flatten, Dropout, MaxPooling1D, Dense, BatchNormalization, Activation, UpSampling1D, Concatenate
from keras.callbacks import LearningRateScheduler
from keras.initializers import TruncatedNormal
from keras.engine.topology import Layer
from keras.layers import Lambda
from keras.models import Model, load_model
from keras import backend as K
from keras import optimizers
from keras import layers
import numpy as np
import pickle
import h5py
import math

class AMSoftmax(Layer):

    def __init__(self, output_dim, s, m, **kwargs):
        self.output_dim = output_dim
        self.s = s
        self.m = m
        super(AMSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[0][-1], self.output_dim),
                                      initializer=TruncatedNormal(mean=0.0, stddev=1.0),
                                      trainable=True)

        self.bias = self.add_weight(name='bias',
    	 			    shape=(self.output_dim, ),
				    initializer=TruncatedNormal(mean=0.0, stddev=1.0),
 		                    trainable=True)

        super(AMSoftmax, self).build(input_shape)
        
    def call(self, inputs): 
        x = inputs[0]
        y = inputs[1]
        kernel_norm = K.l2_normalize(self.kernel, 0)
        cos_theta = K.dot(x, kernel_norm)
        cos_theta = K.clip(cos_theta, -1,1) # for numerical steady
        cos_theta = K.bias_add(cos_theta, self.bias, data_format='channels_last')
        phi = cos_theta - self.m
        adjust_theta = self.s * K.tf.where(K.tf.equal(y, 1), phi, cos_theta)
        return K.softmax(adjust_theta)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.output_dim)

hf = h5py.File('/scratch/kjakkala/neuralwave/data/pca_data.h5', 'r')
X_train = np.expand_dims(hf.get('X_train'), axis=-1)
X_test = np.expand_dims(hf.get('X_test'), axis=-1)
y_train = np.eye(30)[hf.get('y_train')]
y_test = np.eye(30)[hf.get('y_test')]
hf.close()

lr = 1e-6
epochs=50000
margin=0.35

labels = Input(shape=(30,), name='labels')
base_model = load_model("/users/kjakkala/neuralwave/weights/resnet25_softmax.h5")
x = AMSoftmax(30, s=30, m=margin)([base_model.layers[-2].output, labels])
model = Model(inputs=[base_model.input, labels], outputs=x)
model.summary()

model.load_weights("/users/kjakkala/neuralwave/weights/resnet25_am_softmax.h5", by_name=True)
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr), metrics=['acc'])
history = model.fit(x=[X_train, y_train], y=y_train, epochs=epochs, validation_data=([X_test, y_test], y_test), verbose=2)
model.save_weights("/users/kjakkala/neuralwave/weights/resnet25_am_softmax.h5")

fileObject = open("/users/kjakkala/resnet25_softmax_am1.pkl", 'wb')
pickle.dump(history.history,fileObject)
fileObject.close()

