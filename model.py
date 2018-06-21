import keras.backend as K
from keras.layers import Input, Conv2D, Add, Concatenate
from keras.models import Model
from keras.utils import plot_model

import utils
from config import img_size, channel, kernel


def build_model(num_layers=80, feature_size=64, scaling_factor=1.0):
    input_tensor = Input(shape=(img_size, img_size, channel))

    # One convolution before res blocks and to convert to required feature depth
    x = Conv2D(feature_size, (kernel, kernel), activation='relu', padding='same', name='conv1')(input_tensor)

    temp = K.variable((3,))
    for i in [2, 3, 4]:
        temp[i] = x
        for _ in [1, 2]:
            temp[i] = Conv2D(feature_size, (5, 5), activation='relu', padding='same', name='conv1')(temp[i])
            temp[i] = Conv2D(feature_size, (5, 5), padding='same', name='conv1')(temp[i])
    x = Concatenate([temp[2], temp[3], temp[4]])

    # Store the output of the first convolution to add later
    conv_1 = x

    # Add the residual blocks to the model
    for i in range(num_layers):
        x = utils.res_block(x, feature_size, scale=scaling_factor)

    x = Conv2D(feature_size, (kernel, kernel), padding='same')(x)
    x = Add()([x, conv_1])

    # Upsample output of the convolution
    x2 = utils.upsample(x, 2, feature_size)
    x3 = utils.upsample(x, 3, feature_size)
    x4 = utils.upsample(x, 4, feature_size)

    outputs = [x2, x3, x4]

    model = Model(inputs=input_tensor, outputs=outputs, name="EDRN")
    return model


if __name__ == '__main__':
    m = build_model()
    print(m.summary())
    plot_model(m, to_file='model.svg', show_layer_names=True, show_shapes=True)
    K.clear_session()
