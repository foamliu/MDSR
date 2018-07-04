# import the necessary packages
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np

from model import build_model
from config import img_size, max_scale
from data_generator import random_crop, preprocess_input

if __name__ == '__main__':
    model_weights_path = 'models/model.16-21.4264.hdf5'
    model = build_model()
    model.load_weights(model_weights_path)

    print(model.summary())

    image_folder = '/mnt/code/ImageNet-Downloader/image/resized'
    names_file = 'valid_names.txt'
    with open(names_file, 'r') as f:
        names = f.read().splitlines()

    samples = random.sample(names, 10)

    h, w = img_size * max_scale, img_size * max_scale

    for i in range(len(samples)):
        image_name = samples[i]
        filename = os.path.join(image_folder, image_name)
        print('Start processing image: {}'.format(filename))
        image_bgr = cv.imread(filename)
        y = random_crop(image_bgr)

        x = cv.resize(y, (img_size, img_size), cv.INTER_CUBIC)
        input = x.copy()
        input_x4 = cv.resize(input, (img_size * max_scale, img_size * max_scale), cv.INTER_CUBIC)

        x = preprocess_input(x.astype(np.float32))
        x_test = np.empty((1, img_size, img_size, 3), dtype=np.float32)
        x_test[0] = x
        out = model.predict(x_test)

        out_x2 = out[0][0]
        out_x2 = np.clip(out_x2, 0.0, 255.0)
        out_x2 = out_x2.astype(np.uint8)

        out_x3 = out[1][0]
        out_x3 = np.clip(out_x3, 0.0, 255.0)
        out_x3 = out_x3.astype(np.uint8)

        out_x4 = out[2][0]
        out_x4 = np.clip(out_x4, 0.0, 255.0)
        out_x4 = out_x4.astype(np.uint8)

        if not os.path.exists('images'):
            os.makedirs('images')

        cv.imwrite('images/{}_input.png'.format(i), input)
        cv.imwrite('images/{}_input_x4.png'.format(i), input_x4)
        cv.imwrite('images/{}_gt.png'.format(i), y)
        cv.imwrite('images/{}_out_x2.png'.format(i), out_x2)
        cv.imwrite('images/{}_out_x3.png'.format(i), out_x3)
        cv.imwrite('images/{}_out_x4.png'.format(i), out_x4)

    K.clear_session()
