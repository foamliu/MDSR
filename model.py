import keras.backend as K
from keras.layers import Input, Conv2D, add
from keras.models import Model
from keras.utils import plot_model

import utils
from config import img_size, channel, kernel


def build_model(num_layers=80, feature_size=64, scaling_factor=1.0):
    input_tensor = Input(shape=(img_size, img_size, channel))

    # One convolution before res blocks and to convert to required feature depth
    x = Conv2D(feature_size, (kernel, kernel), activation='relu', padding='same')(input_tensor)

    for _ in [1, 2]:
        conv_x2 = Conv2D(feature_size, (5, 5), activation='relu', padding='same')(x)
        conv_x2 = Conv2D(feature_size, (5, 5), padding='same')(conv_x2)

    for _ in [1, 2]:
        conv_x3 = Conv2D(feature_size, (5, 5), activation='relu', padding='same')(x)
        conv_x3 = Conv2D(feature_size, (5, 5), padding='same')(conv_x3)

    for _ in [1, 2]:
        conv_x4 = Conv2D(feature_size, (5, 5), activation='relu', padding='same')(x)
        conv_x4 = Conv2D(feature_size, (5, 5), padding='same')(conv_x4)

    # Add the residual blocks to the model
    for i in range(num_layers):
        x = utils.res_block(x, feature_size, scale=scaling_factor)

    x = Conv2D(feature_size, (kernel, kernel), padding='same')(x)

    # Upsample output of the convolution
    x2 = utils.upsample(add([x, conv_x2]), 2, feature_size)
    x3 = utils.upsample(add([x, conv_x3]), 3, feature_size)
    x4 = utils.upsample(add([x, conv_x4]), 4, feature_size)

    outputs = [x2, x3, x4]

    model = Model(inputs=input_tensor, outputs=outputs, name="MDSR")
    return model


if __name__ == '__main__':
    m = build_model()
    print(m.summary())
    plot_model(m, to_file='model.svg', show_layer_names=True, show_shapes=True)
    K.clear_session()
