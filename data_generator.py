import os
import random
from random import shuffle

import cv2 as cv
import numpy as np
from keras.utils import Sequence
import imutils
from config import batch_size, img_size, channel

image_folder = '/mnt/code/ImageNet-Downloader/image/resized'


def random_crop(image_bgr, scale):
    full_size = image_bgr.shape[0]
    y_size = img_size * scale
    u = random.randint(0, full_size - y_size)
    v = random.randint(0, full_size - y_size)
    y = image_bgr[v:v + y_size, u:u + y_size]
    return y


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


class DataGenSequence(Sequence):
    def __init__(self, usage, scale):
        self.usage = usage
        self.scale = scale

        if usage == 'train':
            names_file = 'train_names.txt'
        else:
            names_file = 'valid_names.txt'

        with open(names_file, 'r') as f:
            self.names = f.read().splitlines()

        np.random.shuffle(self.names)

    def __len__(self):
        return int(np.ceil(len(self.names) / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        out_img_rows, out_img_cols = img_size * self.scale, img_size * self.scale

        length = min(batch_size, (len(self.names) - i))
        batch_x = np.empty((length, img_size, img_size, channel), dtype=np.float32)
        batch_y = np.empty((length, out_img_rows, out_img_cols, channel), dtype=np.float32)

        for i_batch in range(length):
            name = self.names[i]
            filename = os.path.join(image_folder, name)
            # b: 0 <=b<=255, g: 0 <=g<=255, r: 0 <=r<=255.
            image_bgr = cv.imread(filename)

            y = random_crop(image_bgr)

            if np.random.random_sample() > 0.5:
                y = np.fliplr(y)

            angle = random.choice((0, 90, 180, 270))
            y = imutils.rotate_bound(y, angle)

            x = cv.resize(y, (img_size, img_size), cv.INTER_CUBIC)

            batch_x[i_batch, :, :] = preprocess_input(x.astype(np.float32))
            batch_y[i_batch, :, :] = y

            i += 1

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.names)


def train_gen(scale):
    return DataGenSequence('train', scale)


def valid_gen(scale):
    return DataGenSequence('valid', scale)


def split_data():
    names = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]

    num_samples = len(names)  # 1341430
    print('num_samples: ' + str(num_samples))

    num_train_samples = int(num_samples * 0.992)
    print('num_train_samples: ' + str(num_train_samples))
    num_valid_samples = num_samples - num_train_samples
    print('num_valid_samples: ' + str(num_valid_samples))
    valid_names = random.sample(names, num_valid_samples)
    train_names = [n for n in names if n not in valid_names]
    shuffle(valid_names)
    shuffle(train_names)

    # with open('names.txt', 'w') as file:
    #     file.write('\n'.join(names))

    with open('valid_names.txt', 'w') as file:
        file.write('\n'.join(valid_names))

    with open('train_names.txt', 'w') as file:
        file.write('\n'.join(train_names))


if __name__ == '__main__':
    split_data()
