import multiprocessing

import cv2 as cv
import keras.backend as K
import tensorflow as tf
from keras.layers import Conv2D, add, Lambda
from tensorflow.python.client import device_lib


def custom_loss(y_true, y_pred):
    diff = y_pred - y_true
    return K.mean(K.sqrt(K.square(diff) + K.epsilon()))


# getting the number of GPUs
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# getting the number of CPUs
def get_available_cpus():
    return multiprocessing.cpu_count()


"""
Creates a convolutional residual block
as defined in the paper. More on
this inside EDSR.py
input_tensor: input to pass through the residual block
features: number of features to compute
stride: convolution stride
"""


def res_block(input_tensor, features=64, kernel=3):
    x = Conv2D(features, (kernel, kernel), activation='relu', padding='same')(input_tensor)
    x = Conv2D(features, (kernel, kernel), padding='same')(x)
    return add([input_tensor, x])


"""
Method to upscale an image using
conv2d transpose. Based on upscaling
method defined in the paper
x: input to be upscaled
scale: scale increase of upsample
features: number of features to compute
activation: activation function
"""


def upsample(x, scale=2, features=64, kernel=3):
    assert scale in [2, 3, 4]
    x = Conv2D(features, (kernel, kernel), padding='same')(x)
    if scale == 2:
        ps_features = 3 * (scale ** 2)
        x = Conv2D(ps_features, (kernel, kernel), padding='same')(x)
        x = PS(x, 2, color=True)
    elif scale == 3:
        ps_features = 3 * (scale ** 2)
        x = Conv2D(ps_features, (kernel, kernel), padding='same')(x)
        x = PS(x, 3, color=True)
    elif scale == 4:
        ps_features = 3 * (2 ** 2)
        for i in range(2):
            x = Conv2D(ps_features, (kernel, kernel), padding='same')(x)
            x = PS(x, 2, color=True)
    return x


"""
Borrowed from https://github.com/tetrachrome/subpixel
Used for subpixel phase shifting after deconv operations
"""


def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    bsize = K.shape(I)[0]  # Handling Dimension(None) type for undefined batch dim
    X = Lambda(lambda x: tf.reshape(x, (bsize, a, b, r, r)))(I)
    X = Lambda(lambda x: tf.transpose(x, (0, 1, 2, 4, 3)))(X)  # bsize, a, b, 1, 1
    X = Lambda(lambda x: tf.split(x, a, 1))(X)  # a, [bsize, b, r, r]
    X = Lambda(lambda X: tf.concat([tf.squeeze(x, axis=1) for x in X], 2))(X)  # bsize, b, a*r, r
    X = Lambda(lambda x: tf.split(x, b, 1))(X)  # b, [bsize, a*r, r]
    X = Lambda(lambda X: tf.concat([tf.squeeze(x, axis=1) for x in X], 2))(X)  # bsize, a*r, b*r
    return Lambda(lambda x: tf.reshape(x, (bsize, a * r, b * r, 1)))(X)


"""
Borrowed from https://github.com/tetrachrome/subpixel
Used for subpixel phase shifting after deconv operations
"""


def PS(X, r, color=False):
    if color:
        Xc = Lambda(lambda x: tf.split(x, 3, 3))(X)
        X = Lambda(lambda xc: tf.concat([_phase_shift(x, r) for x in xc], 3))(Xc)
    else:
        X = _phase_shift(X, r)
    return X


"""
Tensorflow log base 10.
Found here: https://github.com/tensorflow/tensorflow/issues/1666
"""


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)
