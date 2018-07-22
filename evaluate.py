import json
import os

import cv2 as cv
import keras.backend as K
import numpy as np
from tqdm import tqdm

from config import img_size, image_folder, max_scale, eval_path
from model import build_model
from utils import random_crop, preprocess_input, psnr

if __name__ == '__main__':
    model_weights_path = 'models/model.16-21.4264.hdf5'
    model = build_model()
    model.load_weights(model_weights_path)

    names_file = 'valid_names.txt'
    with open(names_file, 'r') as f:
        names = f.read().splitlines()

    h, w = img_size * max_scale, img_size * max_scale

    total_psnr_x2 = 0
    total_psnr_x3 = 0
    total_psnr_x4 = 0
    total_bicubic_x2 = 0
    total_bicubic_x3 = 0
    total_bicubic_x4 = 0

    for i in tqdm(range(len(names))):
        name = names[i]
        filename = os.path.join(image_folder, name)
        image_bgr = cv.imread(filename)
        gt = random_crop(image_bgr)

        x = cv.resize(gt, (img_size, img_size), cv.INTER_CUBIC)
        input = x.copy()

        x = preprocess_input(x.astype(np.float32))
        x_test = np.empty((1, img_size, img_size, 3), dtype=np.float32)
        x_test[0] = x
        out = model.predict(x_test)

        out_x2 = out[0][0]
        out_x2 = np.clip(out_x2, 0.0, 255.0)
        out_x2 = out_x2.astype(np.uint8)
        bicubic_x2 = cv.resize(input, (img_size * 2, img_size * 2), cv.INTER_CUBIC)
        gt_x2 = cv.resize(gt, (img_size * 2, img_size * 2), cv.INTER_CUBIC)
        total_psnr_x2 += psnr(out_x2, gt_x2)
        total_bicubic_x2 += psnr(bicubic_x2, gt_x2)

        out_x3 = out[1][0]
        out_x3 = np.clip(out_x3, 0.0, 255.0)
        out_x3 = out_x3.astype(np.uint8)
        bicubic_x3 = cv.resize(input, (img_size * 3, img_size * 3), cv.INTER_CUBIC)
        gt_x3 = cv.resize(gt, (img_size * 3, img_size * 3), cv.INTER_CUBIC)
        total_psnr_x3 += psnr(out_x3, gt_x3)
        total_bicubic_x3 += psnr(bicubic_x3, gt_x3)

        out_x4 = out[2][0]
        out_x4 = np.clip(out_x4, 0.0, 255.0)
        out_x4 = out_x4.astype(np.uint8)
        bicubic_x4 = cv.resize(input, (img_size * 4, img_size * 4), cv.INTER_CUBIC)
        gt_x4 = gt
        total_psnr_x4 += psnr(out_x4, gt_x4)
        total_bicubic_x3 += psnr(bicubic_x2, gt_x3)

    psnr_avg_x2 = total_psnr_x2 / len(names)
    psnr_avg_x3 = total_psnr_x3 / len(names)
    psnr_avg_x4 = total_psnr_x4 / len(names)
    bicubic_avg_x2 = total_bicubic_x2 / len(names)
    bicubic_avg_x3 = total_bicubic_x3 / len(names)
    bicubic_avg_x4 = total_bicubic_x3 / len(names)

    print('PSNRx2(avg): {0:.5f}'.format(psnr_avg_x2))
    print('PSNRx3(avg): {0:.5f}'.format(psnr_avg_x3))
    print('PSNRx4(avg): {0:.5f}'.format(psnr_avg_x4))
    print('Bicubicx2(avg): {0:.5f}'.format(bicubic_avg_x2))
    print('Bicubicx3(avg): {0:.5f}'.format(bicubic_avg_x3))
    print('Bicubicx4(avg): {0:.5f}'.format(bicubic_avg_x4))

    if os.path.isfile(eval_path):
        with open(eval_path) as file:
            eval_result = json.load(file)
    else:
        eval_result = {}
    eval_result['psnr_avg_x2'] = psnr_avg_x2
    eval_result['psnr_avg_x3'] = psnr_avg_x3
    eval_result['psnr_avg_x4'] = psnr_avg_x4
    eval_result['bicubic_avg_x2'] = bicubic_avg_x2
    eval_result['bicubic_avg_x3'] = bicubic_avg_x3
    eval_result['bicubic_avg_x4'] = bicubic_avg_x4
    with open(eval_path, 'w') as file:
        json.dump(eval_result, file)

    K.clear_session()
