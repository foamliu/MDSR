# -*- coding: utf-8 -*-
import json

from config import eval_path

if __name__ == '__main__':
    with open(eval_path, 'r', encoding="utf-8") as file:
        eval_result = json.load(file)

    with open('README.t', 'r', encoding="utf-8") as file:
        template = file.readlines()

    template = ''.join(template)

    for i in range(10):
        template = template.replace('$(psnr_{}_x2)'.format(i), '{0:.5f}'.format(eval_result['psnr_list_x2'][i]))
        template = template.replace('$(psnr_{}_x3)'.format(i), '{0:.5f}'.format(eval_result['psnr_list_x3'][i]))
        template = template.replace('$(psnr_{}_x4)'.format(i), '{0:.5f}'.format(eval_result['psnr_list_x4'][i]))
        template = template.replace('$(psnr_gt_{}_x4)'.format(i), str(eval_result['psnr_list_gt_x4'][i]))

    template = template.replace('$(psnr_avg_x2)', '{0:.5f} dB'.format(eval_result['psnr_avg_x2']))
    template = template.replace('$(psnr_avg_x3)', '{0:.5f} dB'.format(eval_result['psnr_avg_x3']))
    template = template.replace('$(psnr_avg_x4)', '{0:.5f} dB'.format(eval_result['psnr_avg_x4']))

    with open('README.md', 'w', encoding="utf-8") as file:
        file.write(template)
