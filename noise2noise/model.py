from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Add, PReLU, Conv2DTranspose, Concatenate, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Conv2D, Activation, ZeroPadding2D
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import tensorflow as tf

eps = 1.1e-5

class L0Loss:
    def __init__(self):
        self.gamma = K.variable(2.)

    def __call__(self):
        def calc_loss(y_true, y_pred):
            loss = K.pow(K.abs(y_true - y_pred) + 1e-8, self.gamma)
            return loss
        return calc_loss

class UpdateAnnealingParameter(Callback):
    def __init__(self, gamma, nb_epochs, verbose=0):
        super(UpdateAnnealingParameter, self).__init__()
        self.gamma = gamma
        self.nb_epochs = nb_epochs
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        new_gamma = 2.0 * (self.nb_epochs - epoch) / self.nb_epochs
        K.set_value(self.gamma, new_gamma)

        if self.verbose > 0:
            print('\nEpoch %05d: UpdateAnnealingParameter reducing gamma to %s.' % (epoch + 1, new_gamma))

def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    y_pred = K.clip(y_pred, 0.0, 1.0)
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))

'''
UNet: code from https://github.com/pietz/unet-keras
U-Net: Convolutional Networks for Biomedical Image Segmentation
(https://arxiv.org/abs/1505.04597)
---
img_shape: (height, width, channels)
out_ch: number of output channels
start_ch: number of channels of the first conv
depth: zero indexed depth of the U-structure
inc_rate: rate at which the conv channels will increase
activation: activation function after convolutions
dropout: amount of dropout in the contracting part
batchnorm: adds Batch Normalization if true
maxpool: use strided conv instead of maxpooling if false
upconv: use transposed conv instead of upsamping + conv if false
residual: add residual connections around each conv block if true
'''
def conv_block(input_tensor, filters, activation_func):
    filters1, filters2, filters3 = filters

    x = Conv2D(filters1, 1, activation=activation_func, padding='same')(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=-1)(x)
    x = Activation(activation_func)(x)

    x = Conv2D(filters2, 3, activation=activation_func, padding='same')(x)
    x = BatchNormalization(epsilon=eps, axis=-1)(x)
    x = Activation(activation_func)(x)

    x = Conv2D(filters3, 1, activation=activation_func, padding='same')(x)
    x = BatchNormalization(epsilon=eps, axis=-1)(x)

    x = Concatenate()([x, input_tensor])
    x = Activation(activation_func)(x)

    return x

def level_block(m, filters, activation_func, depth, inc_rate):
    if depth > 0:
        n = conv_block(m, filters, activation_func)
        m = MaxPooling2D()(n)
        m = level_block(m, [inc_rate*x for x in filters], activation_func, depth - 1, inc_rate)
        m = UpSampling2D()(m)
        m = Conv2D(filters[-1], 2, activation=activation_func, padding='same')(m)
        n = Concatenate()([n, m])
        m = conv_block(n, filters, activation_func)
    else:
        m = conv_block(m, filters, activation_func)

    return m

def get_unet_model(input_channel_num, output_channel_num, filters=[4, 4, 8], activation_func='relu', depth=3, inc_rate=2):
    i = Input(shape=(None, None, input_channel_num))
    o = level_block(i, filters, activation_func, depth, inc_rate)
    o = Conv2D(output_channel_num, 1, activation='sigmoid')(o)
    model = Model(inputs=i, outputs=o)

    return model

if __name__ == '__main__':
    pass
