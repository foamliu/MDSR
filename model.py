import keras.backend as K
from keras.layers import Input, Conv2D, add
from keras.models import Model
from keras.utils import plot_model

import utils
from config import img_size, channel


def build_model(num_layers=80, feature_size=64, scaling_factor=1.0):
    input_tensor = Input(shape=(img_size, img_size, channel))

    # One convolution before res blocks and to convert to required feature depth
    x = Conv2D(feature_size, (3, 3), activation='relu', padding='same')(input_tensor)

    conv_x2 = utils.res_block(x, feature_size, scale=scaling_factor, kernel=5)
    conv_x2 = utils.res_block(conv_x2, feature_size, scale=scaling_factor, kernel=5)
    conv_x3 = utils.res_block(x, feature_size, scale=scaling_factor, kernel=5)
    conv_x3 = utils.res_block(conv_x3, feature_size, scale=scaling_factor, kernel=5)
    conv_x4 = utils.res_block(x, feature_size, scale=scaling_factor, kernel=5)
    conv_x4 = utils.res_block(conv_x4, feature_size, scale=scaling_factor, kernel=5)

    x = add([conv_x2, conv_x3, conv_x4])

    # Add the residual blocks to the model
    for i in range(num_layers):
        x = utils.res_block(x, feature_size, scale=scaling_factor)

    x = Conv2D(feature_size, (3, 3), padding='same')(x)

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
