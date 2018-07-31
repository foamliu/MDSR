# import the necessary packages
import json
import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np

from config import img_size, max_scale, image_folder, eval_path, best_model
from data_generator import random_crop, preprocess_input
from model import build_model
from utils import psnr

if __name__ == '__main__':
    model_weights_path = os.path.join('models', best_model)
    model = build_model()
    model.load_weights(model_weights_path)

    print(model.summary())

    names_file = 'valid_names.txt'
    with open(names_file, 'r') as f:
        names = f.read().splitlines()

    samples = random.sample(names, 10)

    h, w = img_size * max_scale, img_size * max_scale

    psnr_list_x2 = []
    psnr_list_x3 = []
    psnr_list_x4 = []
    psnr_list_input_x4 = []
    psnr_list_gt_x4 = []

    for i in range(len(samples)):
        image_name = samples[i]
        filename = os.path.join(image_folder, image_name)
        print('Start processing image: {}'.format(filename))
        image_bgr = cv.imread(filename)
        gt = random_crop(image_bgr)
        psnr_list_gt_x4.append(psnr(gt, gt))

        x = cv.resize(gt, (img_size, img_size), cv.INTER_CUBIC)
        input = x.copy()
        input_x4 = cv.resize(input, (img_size * max_scale, img_size * max_scale), cv.INTER_CUBIC)
        psnr_list_input_x4.append(psnr(input_x4, gt))

        x = preprocess_input(x.astype(np.float32))
        x_test = np.empty((1, img_size, img_size, 3), dtype=np.float32)
        x_test[0] = x
        out = model.predict(x_test)

        out_x2 = out[0][0]
        out_x2 = np.clip(out_x2, 0.0, 255.0)
        out_x2 = out_x2.astype(np.uint8)
        gt_x2 = cv.resize(gt, (img_size * 2, img_size * 2), cv.INTER_CUBIC)
        psnr_list_x2.append(psnr(out_x2, gt_x2))

        out_x3 = out[1][0]
        out_x3 = np.clip(out_x3, 0.0, 255.0)
        out_x3 = out_x3.astype(np.uint8)
        gt_x3 = cv.resize(gt, (img_size * 3, img_size * 3), cv.INTER_CUBIC)
        psnr_list_x3.append(psnr(out_x3, gt_x3))

        out_x4 = out[2][0]
        out_x4 = np.clip(out_x4, 0.0, 255.0)
        out_x4 = out_x4.astype(np.uint8)
        gt_x4 = cv.resize(gt, (img_size * 4, img_size * 4), cv.INTER_CUBIC)
        psnr_list_x4.append(psnr(out_x4, gt_x4))

        if not os.path.exists('images'):
            os.makedirs('images')

        cv.imwrite('images/{}_input.png'.format(i), input)
        cv.imwrite('images/{}_input_x4.png'.format(i), input_x4)
        cv.imwrite('images/{}_gt.png'.format(i), gt)
        cv.imwrite('images/{}_out_x2.png'.format(i), out_x2)
        cv.imwrite('images/{}_out_x3.png'.format(i), out_x3)
        cv.imwrite('images/{}_out_x4.png'.format(i), out_x4)

    if os.path.isfile(eval_path):
        with open(eval_path) as file:
            eval_result = json.load(file)
    else:
        eval_result = {}
    eval_result['psnr_list_x2'] = psnr_list_x2
    eval_result['psnr_list_x3'] = psnr_list_x3
    eval_result['psnr_list_x4'] = psnr_list_x4
    eval_result['psnr_list_input_x4'] = psnr_list_input_x4
    eval_result['psnr_list_gt_x4'] = psnr_list_gt_x4
    with open(eval_path, 'w') as file:
        json.dump(eval_result, file, indent=4)

    K.clear_session()
