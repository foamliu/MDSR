# 超分辨率网络

![apm](https://img.shields.io/apm/l/vim-mode.svg)

MDSR 的 Keras 实现。

## 原理

请参照论文 [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/pdf/1707.02921.pdf)。

本代码参照了原作者的 Torch 实现：[NTIRE2017](https://github.com/LimBee/NTIRE2017) 和 jmiller656 的 Tensorflow 实现 [EDSR-Tensorflow](https://github.com/jmiller656/EDSR-Tensorflow).

MDSR (多尺度模型。 我们提供尺寸x2，x3，x4的模型):

![image](https://github.com/foamliu/MDSR/raw/master/images/MDSR.png)

## 依赖
- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## 数据集

![image](https://github.com/foamliu/MDSR/raw/master/images/imagenet.png)

按照 [说明](https://github.com/foamliu/ImageNet-Downloader) 下载 ImageNet 数据集。# 超分辨率网络

![apm](https://img.shields.io/apm/l/vim-mode.svg)

MDSR 的 Keras 实现。

## 原理

请参照论文 [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/pdf/1707.02921.pdf)。

本代码参照了原作者的 Torch 实现：[NTIRE2017](https://github.com/LimBee/NTIRE2017) 和 jmiller656 的 Tensorflow 实现 [EDSR-Tensorflow](https://github.com/jmiller656/EDSR-Tensorflow).

MDSR (多尺度模型。 我们提供尺寸x2，x3，x4的模型):

![image](https://github.com/foamliu/MDSR/raw/master/images/MDSR.png)

## 依赖
- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## 数据集

![image](https://github.com/foamliu/MDSR/raw/master/images/imagenet.png)

按照 [说明](https://github.com/foamliu/ImageNet-Downloader) 下载 ImageNet 数据集。


## 如何使用


### 训练
```bash
$ python train.py
```

如果想可视化训练效果，请运行:
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

![image](https://github.com/foamliu/MDSR/raw/master/images/learning_curve.png)

### 演示

下载预训练的 [MDSR模型](https://github.com/foamliu/MDSR/releases/download/v1.0/model.16-21.4264.hdf5)，放入 "models" 目录。然后执行:


```bash
$ python demo.py
```

|x|输入|输入x4|x2输出|x3输出|x4输出|真实x4|
|---|---|---|---|---|---|---|
|图片|![image](https://github.com/foamliu/MDSR/raw/master/images/0_input.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/0_input_x4.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/0_out_x2.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/0_out_x3.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/0_out_x4.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/0_gt.png)|
|PSNR|n/a|n/a|$(psnr_0_x2)|$(psnr_0_x3)|$(psnr_0_x4)|$(psnr_gt_0_x4)|
|图片|![image](https://github.com/foamliu/MDSR/raw/master/images/1_input.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/1_input_x4.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/1_out_x2.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/1_out_x3.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/1_out_x4.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/1_gt.png)|
|PSNR|n/a|n/a|$(psnr_1_x2)|$(psnr_1_x3)|$(psnr_1_x4)|$(psnr_gt_1_x4)|
|图片|![image](https://github.com/foamliu/MDSR/raw/master/images/2_input.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/2_input_x4.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/2_out_x2.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/2_out_x3.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/2_out_x4.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/2_gt.png)|
|PSNR|n/a|n/a|$(psnr_2_x2)|$(psnr_2_x3)|$(psnr_2_x4)|$(psnr_gt_2_x4)|
|图片|![image](https://github.com/foamliu/MDSR/raw/master/images/3_input.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/3_input_x4.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/3_out_x2.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/3_out_x3.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/3_out_x4.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/3_gt.png)|
|PSNR|n/a|n/a|$(psnr_3_x2)|$(psnr_3_x3)|$(psnr_3_x4)|$(psnr_gt_3_x4)|
|图片|![image](https://github.com/foamliu/MDSR/raw/master/images/4_input.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/4_input_x4.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/4_out_x2.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/4_out_x3.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/4_out_x4.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/4_gt.png)|
|PSNR|n/a|n/a|$(psnr_4_x2)|$(psnr_4_x3)|$(psnr_4_x4)|$(psnr_gt_4_x4)|
|图片|![image](https://github.com/foamliu/MDSR/raw/master/images/5_input.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/5_input_x4.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/5_out_x2.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/5_out_x3.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/5_out_x4.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/5_gt.png)|
|PSNR|n/a|n/a|$(psnr_5_x2)|$(psnr_5_x3)|$(psnr_5_x4)|$(psnr_gt_5_x4)|
|图片|![image](https://github.com/foamliu/MDSR/raw/master/images/6_input.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/6_input_x4.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/6_out_x2.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/6_out_x3.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/6_out_x4.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/6_gt.png)|
|PSNR|n/a|n/a|$(psnr_6_x2)|$(psnr_6_x3)|$(psnr_6_x4)|$(psnr_gt_6_x4)|
|图片|![image](https://github.com/foamliu/MDSR/raw/master/images/7_input.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/7_input_x4.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/7_out_x2.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/7_out_x3.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/7_out_x4.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/7_gt.png)|
|PSNR|n/a|n/a|$(psnr_7_x2)|$(psnr_7_x3)|$(psnr_7_x4)|$(psnr_gt_7_x4)|
|图片|![image](https://github.com/foamliu/MDSR/raw/master/images/8_input.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/8_input_x4.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/8_out_x2.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/8_out_x3.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/8_out_x4.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/8_gt.png)|
|PSNR|n/a|n/a|$(psnr_8_x2)|$(psnr_8_x3)|$(psnr_8_x4)|$(psnr_gt_8_x4)|
|图片|![image](https://github.com/foamliu/MDSR/raw/master/images/9_input.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/9_input_x4.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/9_out_x2.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/9_out_x3.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/9_out_x4.png)|![image](https://github.com/foamliu/MDSR/raw/master/images/9_gt.png)|
|PSNR|n/a|n/a|$(psnr_9_x2)|$(psnr_9_x3)|$(psnr_9_x4)|$(psnr_gt_9_x4)|

### 评估

在 4268 张验证集图片上测得 PSNR 并求均值：x2=$(psnr_avg_x2), x3=$(psnr_avg_x3), x4=$(psnr_avg_x4)。

```bash
$ python evaluate.py
```


