# The data
This notebook aims at presenting the process of creating the end-to-end modelling pipeline with the use of [PyPlatypus](https://github.com/maju116/pyplatypus) package.

As starting point, the data is crucial, let's proceed with gathering it hence. We will be working with the satellite images of which the [38-Cloud: Cloud Segmentation in Satellite Images dataset](https://www.kaggle.com/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images) is composed. These are TIF files organized in accordance with the following structure:

38-Cloud_training

----|train_blue

--------|blue_IMAGE_NAME1.TIF

--------|blue_IMAGE_NAME2.TIF

...

--------|blue_IMAGE_NAMEN.TIF

----|train_green

--------|green_IMAGE_NAME1.TIF

--------|green_IMAGE_NAME2.TIF

...

--------|green_IMAGE_NAMEN.TIF

----|train_red

--------|red_IMAGE_NAME1.TIF

--------|red_IMAGE_NAME2.TIF

...

--------|red_IMAGE_NAMEN.TIF

----|train_nir

--------|nir_IMAGE_NAME1.TIF

--------|nir_IMAGE_NAME2.TIF

...

--------|nir_IMAGE_NAMEN.TIF

----|train_gt

--------|gt_IMAGE_NAME1.TIF

--------|gt_IMAGE_NAME2.TIF

...

--------|gt_IMAGE_NAMEN.TIF

Our image will be composed of 4 layers in train_blue, train_green, train_red and train_nir folders. In the train_gt we can fin true segmentation masks.

# The preparation
After downloading the data, unpack it and move to any preferred destination. For this example we will be interested only in 38-Cloud_training subdirectories, thus other files could be put aside. Let's take a look at the exemplary image.


```python
! pip install pyplatypus
```


```python
from pathlib import Path
import os
import pandas as pd
cwd = Path.cwd()
while cwd.stem != "pyplatypus":
    cwd = cwd.parent
os.chdir(cwd)
```

In the first step we will create a config file with the images and masks paths for each instance. 


```python
cloud_path = "examples/data/38-Cloud/38-Cloud_training/train_gt/"
mask_paths = [os.path.join(cloud_path, mp) for mp in os.listdir(cloud_path)]
red_images_paths = [mp.replace('gt', 'red') for mp in mask_paths]
green_images_paths = [mp.replace('gt', 'green') for mp in mask_paths]
blue_images_paths = [mp.replace('gt', 'blue') for mp in mask_paths]
nir_images_paths = [mp.replace('gt', 'nir') for mp in mask_paths]
config_df = pd.DataFrame({
    'images': [r + ';' + g + ';' + b + ';' + n for r, g, b, n in zip(red_images_paths, green_images_paths, blue_images_paths, nir_images_paths)],
    'masks': mask_paths
})
config_df.iloc[0:5040, :].to_csv('examples/30_clouds_train.csv', index=False)
config_df.iloc[5040:6720, :].to_csv('examples/30_clouds_validation.csv', index=False)
config_df.iloc[6720:, :].to_csv('examples/30_clouds_test.csv', index=False)
print("Sample image paths:")
print(config_df.images[0])
print("Sample mask path:")
print(config_df.masks[0])
```

    Sample image paths:
    examples/data/38-Cloud/38-Cloud_training/train_red/red_patch_161_8_by_7_LC08_L1TP_064014_20160420_20170223_01_T1.TIF;examples/data/38-Cloud/38-Cloud_training/train_green/green_patch_161_8_by_7_LC08_L1TP_064014_20160420_20170223_01_T1.TIF;examples/data/38-Cloud/38-Cloud_training/train_blue/blue_patch_161_8_by_7_LC08_L1TP_064014_20160420_20170223_01_T1.TIF;examples/data/38-Cloud/38-Cloud_training/train_nir/nir_patch_161_8_by_7_LC08_L1TP_064014_20160420_20170223_01_T1.TIF
    Sample mask path:
    examples/data/38-Cloud/38-Cloud_training/train_gt/gt_patch_161_8_by_7_LC08_L1TP_064014_20160420_20170223_01_T1.TIF


Let's now inspect the input message that we are to send to PlatypusSolver in order to run it.


```python
import yaml
import json
with open(r"examples/claud_38_config.yaml") as stream:
    config = yaml.safe_load(stream)
    print(yaml.dump(config, indent=4, sort_keys=True))
```

    object_detection: null
    semantic_segmentation:
        data:
            colormap:
            -   - 0
                - 0
                - 0
            -   - 255
                - 255
                - 255
            column_sep: ;
            mode: config_file
            shuffle: false
            subdirs:
            - images
            - masks
            train_path: examples/30_clouds_train.csv
            validation_path: examples/30_clouds_validation.csv
        models:
        -   activation_layer: relu
            augmentation:
                Blur:
                    always_apply: false
                    blur_limit: 7
                    p: 0.5
                Flip:
                    always_apply: false
                    p: 0.5
                ToFloat:
                    always_apply: true
                    max_value: 65536
                    p: 1.0
            batch_normalization: true
            batch_size: 32
            blocks: 4
            callbacks:
                EarlyStopping:
                    mode: max
                    monitor: val_dice_coefficient
                    patience: 15
                ModelCheckpoint:
                    filepath: 38c_u_net.h5
                    mode: max
                    monitor: val_dice_coefficient
                    save_best_only: true
            channels:
            - 1
            - 1
            - 1
            - 1
            deep_supervision: false
            dropout: 0.2
            epochs: 1
            filters: 16
            fine_tuning_path: 38c_u_net.h5
            fit: false
            h_splits: 0
            kernel_initializer: he_normal
            linknet: false
            loss:
                focal loss:
                    gamma: 1
            metrics:
                Dice Coefficient: null
                IOU Coefficient: null
                Tversky coefficient:
                    alpha: 1
            n_class: 2
            name: 38c_u_net
            net_h: 256
            net_w: 256
            optimizer:
                Adam:
                    amsgrad: false
                    beta_1: 0.9
                    beta_2: 0.999
                    epsilon: 1e-07
                    learning_rate: 0.001
            plus_plus: false
            resunet: false
            u_net_conv_block_width: 2
            use_separable_conv2d: true
            use_spatial_droput2d: true
            use_up_sampling2d: false
            w_splits: 0
        -   activation_layer: relu
            augmentation:
                Blur:
                    always_apply: false
                    blur_limit: 7
                    p: 0.5
                Flip:
                    always_apply: false
                    p: 0.5
                ToFloat:
                    always_apply: true
                    max_value: 65536
                    p: 1.0
            batch_normalization: true
            batch_size: 8
            blocks: 2
            callbacks:
                EarlyStopping:
                    mode: max
                    monitor: val_dice_coefficient
                    patience: 15
                ModelCheckpoint:
                    filepath: 38c_u_net_plus_plus.h5
                    mode: max
                    monitor: val_dice_coefficient
                    save_best_only: true
            channels:
            - 1
            - 1
            - 1
            - 1
            deep_supervision: false
            dropout: 0.2
            epochs: 1
            filters: 16
            fine_tuning_path: 38c_u_net_plus_plus.h5
            fit: false
            h_splits: 0
            kernel_initializer: he_normal
            linknet: false
            loss:
                focal loss:
                    gamma: 1
            metrics:
                Dice Coefficient: null
                IOU Coefficient: null
                Tversky coefficient:
                    alpha: 1
            n_class: 2
            name: 38c_u_net_plus_plus
            net_h: 256
            net_w: 256
            optimizer:
                Adam:
                    amsgrad: false
                    beta_1: 0.9
                    beta_2: 0.999
                    epsilon: 1e-07
                    learning_rate: 0.001
            plus_plus: true
            u_net_conv_block_width: 2
            use_separable_conv2d: true
            use_spatial_dropout2d: true
            use_up_sampling2d: true
            w_splits: 0
    


What might have struck you is that the config is organized so that it might potentially tell the Solver to train multiple models while using a complex augmentation pipelines and loss functions coming from the rather large set of ones available within the PyPlatypus framework. For we have spotted that some images are given in negatives, we added InvertImg transformation to the Res-U-Net augmentation.

![u_net.png](u_net.png)

# The model

The models present in the PyPlatypus segmentation submodule are U-Net based.

U-Net was originally developed for biomedical data segmentation. As you can see in the picture above architecture is very similar to autoencoder and it looks like the letter U, hence the name. Model is composed of 2 parts, and each part has some number of convolutional blocks (3 in the image above). Number of blocks will be hyperparameter in our model.

To build a U-Net model in platypus use u_net function. You have to specify:

* Number of convolutional blocks,
* Input image height and width - it need not to be in the form 2^N, as we added the generalizng layer.
* Indicator determining if the input image will be loaded as grayscale or RGB.
* Number of classes - in our case we have only 2 (background and nuclei).
* Additional arguments for CNN such as: number of filters, dropout rate etc.

Hereafter the models' building process is rather straightforward.


```python
from pyplatypus.solvers.platypus_cv_solver import PlatypusSolver


ps = PlatypusSolver(
    config_yaml_path=Path("examples/claud_38_config.yaml")
)
ps.train()
```

    5040 images detected!
    Set 'steps_per_epoch' to: 158
    1680 images detected!
    Set 'steps_per_epoch' to: 53


    2022-10-12 21:24:22.953609: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:975] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2022-10-12 21:24:22.958717: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:975] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2022-10-12 21:24:22.958930: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:975] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2022-10-12 21:24:22.959592: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2022-10-12 21:24:22.960771: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:975] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2022-10-12 21:24:22.961053: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:975] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2022-10-12 21:24:22.961215: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:975] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2022-10-12 21:24:23.405807: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:975] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2022-10-12 21:24:23.406061: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:975] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2022-10-12 21:24:23.406219: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:975] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
    2022-10-12 21:24:23.406336: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1532] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 7025 MB memory:  -> device: 0, name: NVIDIA GeForce GTX 1070, pci bus id: 0000:01:00.0, compute capability: 6.1


    Epoch 1/100


    2022-10-12 21:24:31.707394: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8101


    158/158 [==============================] - 488s 3s/step - loss: 0.1611 - categorical_crossentropy: 0.4626 - dice_coefficient: 0.6246 - iou_coefficient: 0.4574 - tversky_coefficient: 0.5237 - val_loss: 0.2437 - val_categorical_crossentropy: 0.6301 - val_dice_coefficient: 0.5095 - val_iou_coefficient: 0.3418 - val_tversky_coefficient: 0.4059
    Epoch 2/100
    158/158 [==============================] - 466s 3s/step - loss: 0.0857 - categorical_crossentropy: 0.2694 - dice_coefficient: 0.7692 - iou_coefficient: 0.6260 - tversky_coefficient: 0.6845 - val_loss: 0.2125 - val_categorical_crossentropy: 0.5803 - val_dice_coefficient: 0.5372 - val_iou_coefficient: 0.3673 - val_tversky_coefficient: 0.4327
    Epoch 3/100
    158/158 [==============================] - 466s 3s/step - loss: 0.0698 - categorical_crossentropy: 0.2134 - dice_coefficient: 0.8155 - iou_coefficient: 0.6893 - tversky_coefficient: 0.7408 - val_loss: 0.1712 - val_categorical_crossentropy: 0.4942 - val_dice_coefficient: 0.5967 - val_iou_coefficient: 0.4254 - val_tversky_coefficient: 0.4925
    Epoch 4/100
    158/158 [==============================] - 464s 3s/step - loss: 0.0650 - categorical_crossentropy: 0.1940 - dice_coefficient: 0.8323 - iou_coefficient: 0.7136 - tversky_coefficient: 0.7618 - val_loss: 0.1159 - val_categorical_crossentropy: 0.3223 - val_dice_coefficient: 0.7377 - val_iou_coefficient: 0.5849 - val_tversky_coefficient: 0.6467
    Epoch 5/100
    158/158 [==============================] - 465s 3s/step - loss: 0.0598 - categorical_crossentropy: 0.1800 - dice_coefficient: 0.8426 - iou_coefficient: 0.7291 - tversky_coefficient: 0.7750 - val_loss: 0.0621 - val_categorical_crossentropy: 0.2005 - val_dice_coefficient: 0.8209 - val_iou_coefficient: 0.6969 - val_tversky_coefficient: 0.7469
    Epoch 6/100
    158/158 [==============================] - 471s 3s/step - loss: 0.0566 - categorical_crossentropy: 0.1681 - dice_coefficient: 0.8525 - iou_coefficient: 0.7439 - tversky_coefficient: 0.7877 - val_loss: 0.0599 - val_categorical_crossentropy: 0.1784 - val_dice_coefficient: 0.8419 - val_iou_coefficient: 0.7277 - val_tversky_coefficient: 0.7734
    Epoch 7/100
    158/158 [==============================] - 460s 3s/step - loss: 0.0557 - categorical_crossentropy: 0.1633 - dice_coefficient: 0.8568 - iou_coefficient: 0.7505 - tversky_coefficient: 0.7932 - val_loss: 0.0803 - val_categorical_crossentropy: 0.2159 - val_dice_coefficient: 0.8217 - val_iou_coefficient: 0.6983 - val_tversky_coefficient: 0.7480
    Epoch 8/100
    158/158 [==============================] - 461s 3s/step - loss: 0.0542 - categorical_crossentropy: 0.1583 - dice_coefficient: 0.8610 - iou_coefficient: 0.7569 - tversky_coefficient: 0.7986 - val_loss: 0.1105 - val_categorical_crossentropy: 0.2339 - val_dice_coefficient: 0.8279 - val_iou_coefficient: 0.7074 - val_tversky_coefficient: 0.7558
    Epoch 9/100
    158/158 [==============================] - 462s 3s/step - loss: 0.0531 - categorical_crossentropy: 0.1579 - dice_coefficient: 0.8602 - iou_coefficient: 0.7558 - tversky_coefficient: 0.7976 - val_loss: 0.0578 - val_categorical_crossentropy: 0.1710 - val_dice_coefficient: 0.8474 - val_iou_coefficient: 0.7358 - val_tversky_coefficient: 0.7803
    Epoch 10/100
    158/158 [==============================] - 466s 3s/step - loss: 0.0514 - categorical_crossentropy: 0.1497 - dice_coefficient: 0.8675 - iou_coefficient: 0.7671 - tversky_coefficient: 0.8071 - val_loss: 0.0421 - val_categorical_crossentropy: 0.1543 - val_dice_coefficient: 0.8526 - val_iou_coefficient: 0.7437 - val_tversky_coefficient: 0.7871
    Epoch 11/100
    158/158 [==============================] - 464s 3s/step - loss: 0.0497 - categorical_crossentropy: 0.1465 - dice_coefficient: 0.8693 - iou_coefficient: 0.7699 - tversky_coefficient: 0.8095 - val_loss: 0.0583 - val_categorical_crossentropy: 0.1512 - val_dice_coefficient: 0.8704 - val_iou_coefficient: 0.7715 - val_tversky_coefficient: 0.8102
    Epoch 12/100
    158/158 [==============================] - 467s 3s/step - loss: 0.0484 - categorical_crossentropy: 0.1432 - dice_coefficient: 0.8720 - iou_coefficient: 0.7740 - tversky_coefficient: 0.8129 - val_loss: 0.0553 - val_categorical_crossentropy: 0.1691 - val_dice_coefficient: 0.8479 - val_iou_coefficient: 0.7368 - val_tversky_coefficient: 0.7811
    Epoch 13/100
    158/158 [==============================] - 461s 3s/step - loss: 0.0473 - categorical_crossentropy: 0.1387 - dice_coefficient: 0.8756 - iou_coefficient: 0.7798 - tversky_coefficient: 0.8178 - val_loss: 0.0713 - val_categorical_crossentropy: 0.1784 - val_dice_coefficient: 0.8528 - val_iou_coefficient: 0.7446 - val_tversky_coefficient: 0.7876
    Epoch 14/100
    158/158 [==============================] - 465s 3s/step - loss: 0.0484 - categorical_crossentropy: 0.1411 - dice_coefficient: 0.8740 - iou_coefficient: 0.7773 - tversky_coefficient: 0.8157 - val_loss: 0.0624 - val_categorical_crossentropy: 0.1461 - val_dice_coefficient: 0.8801 - val_iou_coefficient: 0.7868 - val_tversky_coefficient: 0.8229
    Epoch 15/100
    158/158 [==============================] - 464s 3s/step - loss: 0.0474 - categorical_crossentropy: 0.1384 - dice_coefficient: 0.8760 - iou_coefficient: 0.7805 - tversky_coefficient: 0.8183 - val_loss: 0.0409 - val_categorical_crossentropy: 0.1461 - val_dice_coefficient: 0.8599 - val_iou_coefficient: 0.7548 - val_tversky_coefficient: 0.7964
    Epoch 16/100
    158/158 [==============================] - 466s 3s/step - loss: 0.0457 - categorical_crossentropy: 0.1344 - dice_coefficient: 0.8788 - iou_coefficient: 0.7848 - tversky_coefficient: 0.8219 - val_loss: 0.0668 - val_categorical_crossentropy: 0.1686 - val_dice_coefficient: 0.8593 - val_iou_coefficient: 0.7543 - val_tversky_coefficient: 0.7959
    Epoch 17/100
    158/158 [==============================] - 468s 3s/step - loss: 0.0432 - categorical_crossentropy: 0.1295 - dice_coefficient: 0.8819 - iou_coefficient: 0.7896 - tversky_coefficient: 0.8259 - val_loss: 0.0441 - val_categorical_crossentropy: 0.1374 - val_dice_coefficient: 0.8720 - val_iou_coefficient: 0.7738 - val_tversky_coefficient: 0.8123
    Epoch 18/100
    158/158 [==============================] - 462s 3s/step - loss: 0.0453 - categorical_crossentropy: 0.1330 - dice_coefficient: 0.8800 - iou_coefficient: 0.7867 - tversky_coefficient: 0.8235 - val_loss: 0.0351 - val_categorical_crossentropy: 0.1080 - val_dice_coefficient: 0.8957 - val_iou_coefficient: 0.8117 - val_tversky_coefficient: 0.8435
    Epoch 19/100
    158/158 [==============================] - 462s 3s/step - loss: 0.0450 - categorical_crossentropy: 0.1316 - dice_coefficient: 0.8812 - iou_coefficient: 0.7886 - tversky_coefficient: 0.8251 - val_loss: 0.0343 - val_categorical_crossentropy: 0.1114 - val_dice_coefficient: 0.8912 - val_iou_coefficient: 0.8044 - val_tversky_coefficient: 0.8375
    Epoch 20/100
    158/158 [==============================] - 464s 3s/step - loss: 0.0437 - categorical_crossentropy: 0.1276 - dice_coefficient: 0.8844 - iou_coefficient: 0.7937 - tversky_coefficient: 0.8293 - val_loss: 0.0571 - val_categorical_crossentropy: 0.1616 - val_dice_coefficient: 0.8570 - val_iou_coefficient: 0.7506 - val_tversky_coefficient: 0.7927
    Epoch 21/100
    158/158 [==============================] - 464s 3s/step - loss: 0.0424 - categorical_crossentropy: 0.1271 - dice_coefficient: 0.8836 - iou_coefficient: 0.7924 - tversky_coefficient: 0.8282 - val_loss: 0.0335 - val_categorical_crossentropy: 0.1111 - val_dice_coefficient: 0.8910 - val_iou_coefficient: 0.8041 - val_tversky_coefficient: 0.8373
    Epoch 22/100
    158/158 [==============================] - 467s 3s/step - loss: 0.0418 - categorical_crossentropy: 0.1233 - dice_coefficient: 0.8873 - iou_coefficient: 0.7982 - tversky_coefficient: 0.8330 - val_loss: 0.0465 - val_categorical_crossentropy: 0.1460 - val_dice_coefficient: 0.8649 - val_iou_coefficient: 0.7626 - val_tversky_coefficient: 0.8029
    Epoch 23/100
    158/158 [==============================] - 462s 3s/step - loss: 0.0401 - categorical_crossentropy: 0.1202 - dice_coefficient: 0.8890 - iou_coefficient: 0.8010 - tversky_coefficient: 0.8353 - val_loss: 0.0386 - val_categorical_crossentropy: 0.1191 - val_dice_coefficient: 0.8869 - val_iou_coefficient: 0.7975 - val_tversky_coefficient: 0.8319
    Epoch 24/100
    158/158 [==============================] - 461s 3s/step - loss: 0.0415 - categorical_crossentropy: 0.1217 - dice_coefficient: 0.8890 - iou_coefficient: 0.8010 - tversky_coefficient: 0.8353 - val_loss: 0.0318 - val_categorical_crossentropy: 0.1118 - val_dice_coefficient: 0.8887 - val_iou_coefficient: 0.8002 - val_tversky_coefficient: 0.8341
    Epoch 25/100
    158/158 [==============================] - 467s 3s/step - loss: 0.0395 - categorical_crossentropy: 0.1186 - dice_coefficient: 0.8903 - iou_coefficient: 0.8031 - tversky_coefficient: 0.8370 - val_loss: 0.0608 - val_categorical_crossentropy: 0.1676 - val_dice_coefficient: 0.8544 - val_iou_coefficient: 0.7466 - val_tversky_coefficient: 0.7894
    Epoch 26/100
    158/158 [==============================] - 468s 3s/step - loss: 0.0425 - categorical_crossentropy: 0.1225 - dice_coefficient: 0.8888 - iou_coefficient: 0.8007 - tversky_coefficient: 0.8350 - val_loss: 0.0368 - val_categorical_crossentropy: 0.1148 - val_dice_coefficient: 0.8897 - val_iou_coefficient: 0.8020 - val_tversky_coefficient: 0.8356
    Epoch 27/100
    158/158 [==============================] - 466s 3s/step - loss: 0.0397 - categorical_crossentropy: 0.1188 - dice_coefficient: 0.8903 - iou_coefficient: 0.8031 - tversky_coefficient: 0.8370 - val_loss: 0.0571 - val_categorical_crossentropy: 0.1521 - val_dice_coefficient: 0.8678 - val_iou_coefficient: 0.7672 - val_tversky_coefficient: 0.8067
    Epoch 28/100
    158/158 [==============================] - 467s 3s/step - loss: 0.0384 - categorical_crossentropy: 0.1148 - dice_coefficient: 0.8932 - iou_coefficient: 0.8079 - tversky_coefficient: 0.8410 - val_loss: 0.0459 - val_categorical_crossentropy: 0.1232 - val_dice_coefficient: 0.8887 - val_iou_coefficient: 0.8005 - val_tversky_coefficient: 0.8342
    Epoch 29/100
    158/158 [==============================] - 469s 3s/step - loss: 0.0386 - categorical_crossentropy: 0.1146 - dice_coefficient: 0.8938 - iou_coefficient: 0.8088 - tversky_coefficient: 0.8418 - val_loss: 0.0450 - val_categorical_crossentropy: 0.1370 - val_dice_coefficient: 0.8732 - val_iou_coefficient: 0.7757 - val_tversky_coefficient: 0.8138
    Epoch 30/100
    158/158 [==============================] - 463s 3s/step - loss: 0.0392 - categorical_crossentropy: 0.1157 - dice_coefficient: 0.8932 - iou_coefficient: 0.8078 - tversky_coefficient: 0.8409 - val_loss: 0.0399 - val_categorical_crossentropy: 0.1239 - val_dice_coefficient: 0.8830 - val_iou_coefficient: 0.7912 - val_tversky_coefficient: 0.8267
    Epoch 31/100
    158/158 [==============================] - 469s 3s/step - loss: 0.0391 - categorical_crossentropy: 0.1161 - dice_coefficient: 0.8926 - iou_coefficient: 0.8069 - tversky_coefficient: 0.8402 - val_loss: 0.0500 - val_categorical_crossentropy: 0.1316 - val_dice_coefficient: 0.8841 - val_iou_coefficient: 0.7931 - val_tversky_coefficient: 0.8282
    Epoch 32/100
    158/158 [==============================] - 467s 3s/step - loss: 0.0369 - categorical_crossentropy: 0.1109 - dice_coefficient: 0.8962 - iou_coefficient: 0.8127 - tversky_coefficient: 0.8449 - val_loss: 0.0359 - val_categorical_crossentropy: 0.1175 - val_dice_coefficient: 0.8865 - val_iou_coefficient: 0.7967 - val_tversky_coefficient: 0.8313
    Epoch 33/100
    158/158 [==============================] - 467s 3s/step - loss: 0.0377 - categorical_crossentropy: 0.1124 - dice_coefficient: 0.8953 - iou_coefficient: 0.8112 - tversky_coefficient: 0.8437 - val_loss: 0.0462 - val_categorical_crossentropy: 0.1322 - val_dice_coefficient: 0.8798 - val_iou_coefficient: 0.7861 - val_tversky_coefficient: 0.8224
    5040 images detected!
    Set 'steps_per_epoch' to: 630
    1680 images detected!
    Set 'steps_per_epoch' to: 210
    Epoch 1/100
      6/630 [..............................] - ETA: 5:27 - loss: 0.2759 - categorical_crossentropy: 0.6746 - tversky_coefficient: 0.3923 - dice_coefficient: 0.4947 - iou_coefficient: 0.3287WARNING:tensorflow:Callback method `on_train_batch_end` is slow compared to the batch time (batch time: 0.1445s vs `on_train_batch_end` time: 0.3203s). Check your callbacks.
    630/630 [==============================] - 478s 755ms/step - loss: 0.1644 - categorical_crossentropy: 0.4454 - tversky_coefficient: 0.5461 - dice_coefficient: 0.6456 - iou_coefficient: 0.4801 - val_loss: 0.1344 - val_categorical_crossentropy: 0.3807 - val_tversky_coefficient: 0.5916 - val_dice_coefficient: 0.6877 - val_iou_coefficient: 0.5274
    Epoch 2/100
    630/630 [==============================] - 479s 760ms/step - loss: 0.1186 - categorical_crossentropy: 0.3236 - tversky_coefficient: 0.6484 - dice_coefficient: 0.7378 - iou_coefficient: 0.5871 - val_loss: 0.1296 - val_categorical_crossentropy: 0.2962 - val_tversky_coefficient: 0.6929 - val_dice_coefficient: 0.7744 - val_iou_coefficient: 0.6380
    Epoch 3/100
    630/630 [==============================] - 476s 755ms/step - loss: 0.1007 - categorical_crossentropy: 0.2786 - tversky_coefficient: 0.6878 - dice_coefficient: 0.7713 - iou_coefficient: 0.6304 - val_loss: 0.0859 - val_categorical_crossentropy: 0.2150 - val_tversky_coefficient: 0.7546 - val_dice_coefficient: 0.8255 - val_iou_coefficient: 0.7076
    Epoch 4/100
    630/630 [==============================] - 479s 760ms/step - loss: 0.0881 - categorical_crossentropy: 0.2476 - tversky_coefficient: 0.7153 - dice_coefficient: 0.7940 - iou_coefficient: 0.6611 - val_loss: 0.0779 - val_categorical_crossentropy: 0.2150 - val_tversky_coefficient: 0.7447 - val_dice_coefficient: 0.8179 - val_iou_coefficient: 0.6957
    Epoch 5/100
    630/630 [==============================] - 479s 760ms/step - loss: 0.0808 - categorical_crossentropy: 0.2284 - tversky_coefficient: 0.7330 - dice_coefficient: 0.8083 - iou_coefficient: 0.6813 - val_loss: 0.0545 - val_categorical_crossentropy: 0.1739 - val_tversky_coefficient: 0.7740 - val_dice_coefficient: 0.8414 - val_iou_coefficient: 0.7295
    Epoch 6/100
    630/630 [==============================] - 477s 757ms/step - loss: 0.0737 - categorical_crossentropy: 0.2102 - tversky_coefficient: 0.7498 - dice_coefficient: 0.8219 - iou_coefficient: 0.7006 - val_loss: 0.0537 - val_categorical_crossentropy: 0.1662 - val_tversky_coefficient: 0.7840 - val_dice_coefficient: 0.8493 - val_iou_coefficient: 0.7413
    Epoch 7/100
    630/630 [==============================] - 480s 761ms/step - loss: 0.0736 - categorical_crossentropy: 0.2090 - tversky_coefficient: 0.7512 - dice_coefficient: 0.8230 - iou_coefficient: 0.7023 - val_loss: 0.0602 - val_categorical_crossentropy: 0.1953 - val_tversky_coefficient: 0.7513 - val_dice_coefficient: 0.8232 - val_iou_coefficient: 0.7032
    Epoch 8/100
    630/630 [==============================] - 479s 761ms/step - loss: 0.0681 - categorical_crossentropy: 0.1959 - tversky_coefficient: 0.7631 - dice_coefficient: 0.8324 - iou_coefficient: 0.7160 - val_loss: 0.0478 - val_categorical_crossentropy: 0.1581 - val_tversky_coefficient: 0.7882 - val_dice_coefficient: 0.8526 - val_iou_coefficient: 0.7461
    Epoch 9/100
    630/630 [==============================] - 478s 758ms/step - loss: 0.0619 - categorical_crossentropy: 0.1815 - tversky_coefficient: 0.7757 - dice_coefficient: 0.8425 - iou_coefficient: 0.7306 - val_loss: 0.1265 - val_categorical_crossentropy: 0.2804 - val_tversky_coefficient: 0.7118 - val_dice_coefficient: 0.7900 - val_iou_coefficient: 0.6593
    Epoch 10/100
    630/630 [==============================] - 479s 761ms/step - loss: 0.0624 - categorical_crossentropy: 0.1811 - tversky_coefficient: 0.7771 - dice_coefficient: 0.8434 - iou_coefficient: 0.7323 - val_loss: 0.0444 - val_categorical_crossentropy: 0.1486 - val_tversky_coefficient: 0.7978 - val_dice_coefficient: 0.8601 - val_iou_coefficient: 0.7574
    Epoch 11/100
    630/630 [==============================] - 478s 758ms/step - loss: 0.0591 - categorical_crossentropy: 0.1726 - tversky_coefficient: 0.7849 - dice_coefficient: 0.8496 - iou_coefficient: 0.7414 - val_loss: 0.0609 - val_categorical_crossentropy: 0.2044 - val_tversky_coefficient: 0.7408 - val_dice_coefficient: 0.8152 - val_iou_coefficient: 0.6907
    Epoch 12/100
    630/630 [==============================] - 479s 759ms/step - loss: 0.0559 - categorical_crossentropy: 0.1641 - tversky_coefficient: 0.7931 - dice_coefficient: 0.8559 - iou_coefficient: 0.7513 - val_loss: 0.0893 - val_categorical_crossentropy: 0.2556 - val_tversky_coefficient: 0.7042 - val_dice_coefficient: 0.7850 - val_iou_coefficient: 0.6495
    Epoch 13/100
    630/630 [==============================] - 479s 760ms/step - loss: 0.0560 - categorical_crossentropy: 0.1631 - tversky_coefficient: 0.7946 - dice_coefficient: 0.8571 - iou_coefficient: 0.7530 - val_loss: 0.0505 - val_categorical_crossentropy: 0.1875 - val_tversky_coefficient: 0.7519 - val_dice_coefficient: 0.8242 - val_iou_coefficient: 0.7034
    Epoch 14/100
    630/630 [==============================] - 479s 760ms/step - loss: 0.0527 - categorical_crossentropy: 0.1566 - tversky_coefficient: 0.7996 - dice_coefficient: 0.8610 - iou_coefficient: 0.7588 - val_loss: 0.1904 - val_categorical_crossentropy: 0.3591 - val_tversky_coefficient: 0.6852 - val_dice_coefficient: 0.7681 - val_iou_coefficient: 0.6297
    Epoch 15/100
    630/630 [==============================] - 478s 758ms/step - loss: 0.0535 - categorical_crossentropy: 0.1571 - tversky_coefficient: 0.8001 - dice_coefficient: 0.8613 - iou_coefficient: 0.7596 - val_loss: 0.0447 - val_categorical_crossentropy: 0.1745 - val_tversky_coefficient: 0.7635 - val_dice_coefficient: 0.8336 - val_iou_coefficient: 0.7166
    Epoch 16/100
    630/630 [==============================] - 480s 762ms/step - loss: 0.0543 - categorical_crossentropy: 0.1577 - tversky_coefficient: 0.8001 - dice_coefficient: 0.8612 - iou_coefficient: 0.7596 - val_loss: 0.0471 - val_categorical_crossentropy: 0.1430 - val_tversky_coefficient: 0.8086 - val_dice_coefficient: 0.8684 - val_iou_coefficient: 0.7704
    Epoch 17/100
    630/630 [==============================] - 478s 759ms/step - loss: 0.0529 - categorical_crossentropy: 0.1548 - tversky_coefficient: 0.8025 - dice_coefficient: 0.8631 - iou_coefficient: 0.7625 - val_loss: 0.0587 - val_categorical_crossentropy: 0.1490 - val_tversky_coefficient: 0.8149 - val_dice_coefficient: 0.8731 - val_iou_coefficient: 0.7780
    Epoch 18/100
    630/630 [==============================] - 480s 762ms/step - loss: 0.0525 - categorical_crossentropy: 0.1532 - tversky_coefficient: 0.8041 - dice_coefficient: 0.8644 - iou_coefficient: 0.7644 - val_loss: 0.0485 - val_categorical_crossentropy: 0.1377 - val_tversky_coefficient: 0.8183 - val_dice_coefficient: 0.8758 - val_iou_coefficient: 0.7820
    Epoch 19/100
    630/630 [==============================] - 478s 758ms/step - loss: 0.0492 - categorical_crossentropy: 0.1465 - tversky_coefficient: 0.8097 - dice_coefficient: 0.8688 - iou_coefficient: 0.7709 - val_loss: 0.0393 - val_categorical_crossentropy: 0.1206 - val_tversky_coefficient: 0.8313 - val_dice_coefficient: 0.8858 - val_iou_coefficient: 0.7976
    Epoch 20/100
    630/630 [==============================] - 479s 760ms/step - loss: 0.0466 - categorical_crossentropy: 0.1389 - tversky_coefficient: 0.8174 - dice_coefficient: 0.8746 - iou_coefficient: 0.7801 - val_loss: 0.1017 - val_categorical_crossentropy: 0.2370 - val_tversky_coefficient: 0.7419 - val_dice_coefficient: 0.8149 - val_iou_coefficient: 0.6933
    Epoch 21/100
    630/630 [==============================] - 479s 760ms/step - loss: 0.0468 - categorical_crossentropy: 0.1388 - tversky_coefficient: 0.8178 - dice_coefficient: 0.8749 - iou_coefficient: 0.7806 - val_loss: 0.0404 - val_categorical_crossentropy: 0.1316 - val_tversky_coefficient: 0.8170 - val_dice_coefficient: 0.8749 - val_iou_coefficient: 0.7803
    Epoch 22/100
    630/630 [==============================] - 478s 758ms/step - loss: 0.0456 - categorical_crossentropy: 0.1353 - tversky_coefficient: 0.8215 - dice_coefficient: 0.8778 - iou_coefficient: 0.7851 - val_loss: 0.0378 - val_categorical_crossentropy: 0.1336 - val_tversky_coefficient: 0.8123 - val_dice_coefficient: 0.8716 - val_iou_coefficient: 0.7742
    Epoch 23/100
    630/630 [==============================] - 478s 758ms/step - loss: 0.0471 - categorical_crossentropy: 0.1375 - tversky_coefficient: 0.8200 - dice_coefficient: 0.8765 - iou_coefficient: 0.7833 - val_loss: 0.0527 - val_categorical_crossentropy: 0.1584 - val_tversky_coefficient: 0.7936 - val_dice_coefficient: 0.8566 - val_iou_coefficient: 0.7528
    Epoch 24/100
    630/630 [==============================] - 477s 757ms/step - loss: 0.0460 - categorical_crossentropy: 0.1373 - tversky_coefficient: 0.8190 - dice_coefficient: 0.8759 - iou_coefficient: 0.7821 - val_loss: 0.0636 - val_categorical_crossentropy: 0.2086 - val_tversky_coefficient: 0.7396 - val_dice_coefficient: 0.8145 - val_iou_coefficient: 0.6891
    Epoch 25/100
    630/630 [==============================] - 478s 759ms/step - loss: 0.0453 - categorical_crossentropy: 0.1345 - tversky_coefficient: 0.8219 - dice_coefficient: 0.8781 - iou_coefficient: 0.7856 - val_loss: 0.0950 - val_categorical_crossentropy: 0.1987 - val_tversky_coefficient: 0.7896 - val_dice_coefficient: 0.8530 - val_iou_coefficient: 0.7488
    Epoch 26/100
    630/630 [==============================] - 478s 758ms/step - loss: 0.0465 - categorical_crossentropy: 0.1362 - tversky_coefficient: 0.8210 - dice_coefficient: 0.8773 - iou_coefficient: 0.7846 - val_loss: 0.0392 - val_categorical_crossentropy: 0.1303 - val_tversky_coefficient: 0.8175 - val_dice_coefficient: 0.8752 - val_iou_coefficient: 0.7810
    Epoch 27/100
    630/630 [==============================] - 477s 757ms/step - loss: 0.0448 - categorical_crossentropy: 0.1320 - tversky_coefficient: 0.8249 - dice_coefficient: 0.8804 - iou_coefficient: 0.7892 - val_loss: 0.0698 - val_categorical_crossentropy: 0.2303 - val_tversky_coefficient: 0.7166 - val_dice_coefficient: 0.7958 - val_iou_coefficient: 0.6629
    Epoch 28/100
    630/630 [==============================] - 478s 759ms/step - loss: 0.0425 - categorical_crossentropy: 0.1279 - tversky_coefficient: 0.8280 - dice_coefficient: 0.8828 - iou_coefficient: 0.7930 - val_loss: 0.0535 - val_categorical_crossentropy: 0.1278 - val_tversky_coefficient: 0.8392 - val_dice_coefficient: 0.8916 - val_iou_coefficient: 0.8075
    Epoch 29/100
    630/630 [==============================] - 477s 757ms/step - loss: 0.0423 - categorical_crossentropy: 0.1252 - tversky_coefficient: 0.8316 - dice_coefficient: 0.8855 - iou_coefficient: 0.7973 - val_loss: 0.0577 - val_categorical_crossentropy: 0.1461 - val_tversky_coefficient: 0.8171 - val_dice_coefficient: 0.8746 - val_iou_coefficient: 0.7810
    Epoch 30/100
    630/630 [==============================] - 479s 760ms/step - loss: 0.0438 - categorical_crossentropy: 0.1292 - tversky_coefficient: 0.8276 - dice_coefficient: 0.8824 - iou_coefficient: 0.7924 - val_loss: 0.0433 - val_categorical_crossentropy: 0.1307 - val_tversky_coefficient: 0.8215 - val_dice_coefficient: 0.8782 - val_iou_coefficient: 0.7861
    Epoch 31/100
    630/630 [==============================] - 480s 762ms/step - loss: 0.0427 - categorical_crossentropy: 0.1259 - tversky_coefficient: 0.8311 - dice_coefficient: 0.8850 - iou_coefficient: 0.7967 - val_loss: 0.0327 - val_categorical_crossentropy: 0.1022 - val_tversky_coefficient: 0.8496 - val_dice_coefficient: 0.8995 - val_iou_coefficient: 0.8199
    Epoch 32/100
    630/630 [==============================] - 477s 757ms/step - loss: 0.0411 - categorical_crossentropy: 0.1229 - tversky_coefficient: 0.8336 - dice_coefficient: 0.8869 - iou_coefficient: 0.7997 - val_loss: 0.0572 - val_categorical_crossentropy: 0.1492 - val_tversky_coefficient: 0.8118 - val_dice_coefficient: 0.8705 - val_iou_coefficient: 0.7746
    Epoch 33/100
    630/630 [==============================] - 481s 763ms/step - loss: 0.0418 - categorical_crossentropy: 0.1246 - tversky_coefficient: 0.8318 - dice_coefficient: 0.8856 - iou_coefficient: 0.7975 - val_loss: 0.0335 - val_categorical_crossentropy: 0.1140 - val_tversky_coefficient: 0.8339 - val_dice_coefficient: 0.8878 - val_iou_coefficient: 0.8006
    Epoch 34/100
    630/630 [==============================] - 479s 759ms/step - loss: 0.0405 - categorical_crossentropy: 0.1208 - tversky_coefficient: 0.8357 - dice_coefficient: 0.8886 - iou_coefficient: 0.8022 - val_loss: 0.0412 - val_categorical_crossentropy: 0.1279 - val_tversky_coefficient: 0.8230 - val_dice_coefficient: 0.8794 - val_iou_coefficient: 0.7876
    Epoch 35/100
    630/630 [==============================] - 478s 759ms/step - loss: 0.0380 - categorical_crossentropy: 0.1147 - tversky_coefficient: 0.8416 - dice_coefficient: 0.8930 - iou_coefficient: 0.8095 - val_loss: 0.0459 - val_categorical_crossentropy: 0.1386 - val_tversky_coefficient: 0.8143 - val_dice_coefficient: 0.8726 - val_iou_coefficient: 0.7773
    Epoch 36/100
    630/630 [==============================] - 481s 763ms/step - loss: 0.0407 - categorical_crossentropy: 0.1206 - tversky_coefficient: 0.8364 - dice_coefficient: 0.8890 - iou_coefficient: 0.8032 - val_loss: 0.0505 - val_categorical_crossentropy: 0.1320 - val_tversky_coefficient: 0.8290 - val_dice_coefficient: 0.8838 - val_iou_coefficient: 0.7951
    Epoch 37/100
    630/630 [==============================] - 478s 759ms/step - loss: 0.0431 - categorical_crossentropy: 0.1272 - tversky_coefficient: 0.8296 - dice_coefficient: 0.8839 - iou_coefficient: 0.7949 - val_loss: 0.0357 - val_categorical_crossentropy: 0.1185 - val_tversky_coefficient: 0.8301 - val_dice_coefficient: 0.8848 - val_iou_coefficient: 0.7961
    Epoch 38/100
    630/630 [==============================] - 479s 760ms/step - loss: 0.0391 - categorical_crossentropy: 0.1171 - tversky_coefficient: 0.8395 - dice_coefficient: 0.8913 - iou_coefficient: 0.8069 - val_loss: 0.0363 - val_categorical_crossentropy: 0.1202 - val_tversky_coefficient: 0.8281 - val_dice_coefficient: 0.8834 - val_iou_coefficient: 0.7937
    Epoch 39/100
    630/630 [==============================] - 481s 763ms/step - loss: 0.0391 - categorical_crossentropy: 0.1171 - tversky_coefficient: 0.8394 - dice_coefficient: 0.8913 - iou_coefficient: 0.8068 - val_loss: 0.0525 - val_categorical_crossentropy: 0.1826 - val_tversky_coefficient: 0.7603 - val_dice_coefficient: 0.8309 - val_iou_coefficient: 0.7131
    Epoch 40/100
    630/630 [==============================] - 477s 757ms/step - loss: 0.0397 - categorical_crossentropy: 0.1181 - tversky_coefficient: 0.8388 - dice_coefficient: 0.8908 - iou_coefficient: 0.8060 - val_loss: 0.0503 - val_categorical_crossentropy: 0.1385 - val_tversky_coefficient: 0.8189 - val_dice_coefficient: 0.8760 - val_iou_coefficient: 0.7830
    Epoch 41/100
    630/630 [==============================] - 475s 754ms/step - loss: 0.0368 - categorical_crossentropy: 0.1114 - tversky_coefficient: 0.8449 - dice_coefficient: 0.8954 - iou_coefficient: 0.8134 - val_loss: 0.0367 - val_categorical_crossentropy: 0.1361 - val_tversky_coefficient: 0.8067 - val_dice_coefficient: 0.8672 - val_iou_coefficient: 0.7677
    Epoch 42/100
    630/630 [==============================] - 476s 756ms/step - loss: 0.0368 - categorical_crossentropy: 0.1107 - tversky_coefficient: 0.8458 - dice_coefficient: 0.8961 - iou_coefficient: 0.8145 - val_loss: 0.0388 - val_categorical_crossentropy: 0.1257 - val_tversky_coefficient: 0.8237 - val_dice_coefficient: 0.8800 - val_iou_coefficient: 0.7884
    Epoch 43/100
    630/630 [==============================] - 477s 757ms/step - loss: 0.0390 - categorical_crossentropy: 0.1153 - tversky_coefficient: 0.8417 - dice_coefficient: 0.8931 - iou_coefficient: 0.8096 - val_loss: 0.0371 - val_categorical_crossentropy: 0.1174 - val_tversky_coefficient: 0.8332 - val_dice_coefficient: 0.8871 - val_iou_coefficient: 0.7999
    Epoch 44/100
    630/630 [==============================] - 477s 757ms/step - loss: 0.0371 - categorical_crossentropy: 0.1115 - tversky_coefficient: 0.8449 - dice_coefficient: 0.8954 - iou_coefficient: 0.8134 - val_loss: 0.0325 - val_categorical_crossentropy: 0.1046 - val_tversky_coefficient: 0.8459 - val_dice_coefficient: 0.8968 - val_iou_coefficient: 0.8155
    Epoch 45/100
    630/630 [==============================] - 498s 789ms/step - loss: 0.0366 - categorical_crossentropy: 0.1097 - tversky_coefficient: 0.8470 - dice_coefficient: 0.8971 - iou_coefficient: 0.8159 - val_loss: 0.0363 - val_categorical_crossentropy: 0.1032 - val_tversky_coefficient: 0.8528 - val_dice_coefficient: 0.9019 - val_iou_coefficient: 0.8239
    Epoch 46/100
    630/630 [==============================] - 486s 770ms/step - loss: 0.0386 - categorical_crossentropy: 0.1147 - tversky_coefficient: 0.8421 - dice_coefficient: 0.8934 - iou_coefficient: 0.8101 - val_loss: 0.0334 - val_categorical_crossentropy: 0.1097 - val_tversky_coefficient: 0.8397 - val_dice_coefficient: 0.8921 - val_iou_coefficient: 0.8079
    Epoch 47/100
    630/630 [==============================] - 488s 774ms/step - loss: 0.0359 - categorical_crossentropy: 0.1072 - tversky_coefficient: 0.8495 - dice_coefficient: 0.8990 - iou_coefficient: 0.8190 - val_loss: 0.0295 - val_categorical_crossentropy: 0.0985 - val_tversky_coefficient: 0.8510 - val_dice_coefficient: 0.9007 - val_iou_coefficient: 0.8217
    Epoch 48/100
    630/630 [==============================] - 491s 778ms/step - loss: 0.0394 - categorical_crossentropy: 0.1165 - tversky_coefficient: 0.8404 - dice_coefficient: 0.8921 - iou_coefficient: 0.8079 - val_loss: 0.0564 - val_categorical_crossentropy: 0.1325 - val_tversky_coefficient: 0.8369 - val_dice_coefficient: 0.8897 - val_iou_coefficient: 0.8049
    Epoch 49/100
    630/630 [==============================] - 491s 779ms/step - loss: 0.0370 - categorical_crossentropy: 0.1106 - tversky_coefficient: 0.8459 - dice_coefficient: 0.8963 - iou_coefficient: 0.8147 - val_loss: 0.0787 - val_categorical_crossentropy: 0.1960 - val_tversky_coefficient: 0.7710 - val_dice_coefficient: 0.8383 - val_iou_coefficient: 0.7267
    Epoch 50/100
    630/630 [==============================] - 488s 774ms/step - loss: 0.0375 - categorical_crossentropy: 0.1119 - tversky_coefficient: 0.8449 - dice_coefficient: 0.8955 - iou_coefficient: 0.8135 - val_loss: 0.0365 - val_categorical_crossentropy: 0.1197 - val_tversky_coefficient: 0.8292 - val_dice_coefficient: 0.8841 - val_iou_coefficient: 0.7951
    Epoch 51/100
    630/630 [==============================] - 540s 857ms/step - loss: 0.0369 - categorical_crossentropy: 0.1099 - tversky_coefficient: 0.8470 - dice_coefficient: 0.8970 - iou_coefficient: 0.8160 - val_loss: 0.0312 - val_categorical_crossentropy: 0.1049 - val_tversky_coefficient: 0.8439 - val_dice_coefficient: 0.8953 - val_iou_coefficient: 0.8129
    Epoch 52/100
    630/630 [==============================] - 515s 818ms/step - loss: 0.0352 - categorical_crossentropy: 0.1069 - tversky_coefficient: 0.8491 - dice_coefficient: 0.8986 - iou_coefficient: 0.8186 - val_loss: 0.0369 - val_categorical_crossentropy: 0.1063 - val_tversky_coefficient: 0.8496 - val_dice_coefficient: 0.8995 - val_iou_coefficient: 0.8199
    Epoch 53/100
    630/630 [==============================] - 538s 854ms/step - loss: 0.0359 - categorical_crossentropy: 0.1069 - tversky_coefficient: 0.8500 - dice_coefficient: 0.8993 - iou_coefficient: 0.8198 - val_loss: 0.0360 - val_categorical_crossentropy: 0.0940 - val_tversky_coefficient: 0.8660 - val_dice_coefficient: 0.9117 - val_iou_coefficient: 0.8402
    Epoch 54/100
    630/630 [==============================] - 538s 853ms/step - loss: 0.0348 - categorical_crossentropy: 0.1046 - tversky_coefficient: 0.8519 - dice_coefficient: 0.9008 - iou_coefficient: 0.8219 - val_loss: 0.0414 - val_categorical_crossentropy: 0.1127 - val_tversky_coefficient: 0.8454 - val_dice_coefficient: 0.8964 - val_iou_coefficient: 0.8150
    Epoch 55/100
    630/630 [==============================] - 554s 878ms/step - loss: 0.0369 - categorical_crossentropy: 0.1095 - tversky_coefficient: 0.8476 - dice_coefficient: 0.8975 - iou_coefficient: 0.8167 - val_loss: 0.0888 - val_categorical_crossentropy: 0.2158 - val_tversky_coefficient: 0.7560 - val_dice_coefficient: 0.8263 - val_iou_coefficient: 0.7094
    Epoch 56/100
    630/630 [==============================] - 535s 849ms/step - loss: 0.0389 - categorical_crossentropy: 0.1144 - tversky_coefficient: 0.8429 - dice_coefficient: 0.8940 - iou_coefficient: 0.8110 - val_loss: 0.0485 - val_categorical_crossentropy: 0.1366 - val_tversky_coefficient: 0.8194 - val_dice_coefficient: 0.8765 - val_iou_coefficient: 0.7835
    Epoch 57/100
    630/630 [==============================] - 553s 878ms/step - loss: 0.0371 - categorical_crossentropy: 0.1106 - tversky_coefficient: 0.8461 - dice_coefficient: 0.8964 - iou_coefficient: 0.8148 - val_loss: 0.0321 - val_categorical_crossentropy: 0.1044 - val_tversky_coefficient: 0.8459 - val_dice_coefficient: 0.8968 - val_iou_coefficient: 0.8153
    Epoch 58/100
    630/630 [==============================] - 604s 959ms/step - loss: 0.0344 - categorical_crossentropy: 0.1048 - tversky_coefficient: 0.8512 - dice_coefficient: 0.9002 - iou_coefficient: 0.8211 - val_loss: 0.0419 - val_categorical_crossentropy: 0.1248 - val_tversky_coefficient: 0.8281 - val_dice_coefficient: 0.8832 - val_iou_coefficient: 0.7941
    Epoch 59/100
    630/630 [==============================] - 719s 1s/step - loss: 0.0351 - categorical_crossentropy: 0.1049 - tversky_coefficient: 0.8518 - dice_coefficient: 0.9006 - iou_coefficient: 0.8219 - val_loss: 0.0508 - val_categorical_crossentropy: 0.1393 - val_tversky_coefficient: 0.8180 - val_dice_coefficient: 0.8753 - val_iou_coefficient: 0.7819
    Epoch 60/100
    630/630 [==============================] - 548s 869ms/step - loss: 0.0348 - categorical_crossentropy: 0.1037 - tversky_coefficient: 0.8532 - dice_coefficient: 0.9017 - iou_coefficient: 0.8236 - val_loss: 0.0360 - val_categorical_crossentropy: 0.1138 - val_tversky_coefficient: 0.8369 - val_dice_coefficient: 0.8900 - val_iou_coefficient: 0.8044
    Epoch 61/100
    630/630 [==============================] - 558s 886ms/step - loss: 0.0359 - categorical_crossentropy: 0.1071 - tversky_coefficient: 0.8496 - dice_coefficient: 0.8990 - iou_coefficient: 0.8192 - val_loss: 0.0273 - val_categorical_crossentropy: 0.0945 - val_tversky_coefficient: 0.8542 - val_dice_coefficient: 0.9031 - val_iou_coefficient: 0.8255
    Epoch 62/100
    630/630 [==============================] - 547s 867ms/step - loss: 0.0349 - categorical_crossentropy: 0.1043 - tversky_coefficient: 0.8526 - dice_coefficient: 0.9013 - iou_coefficient: 0.8229 - val_loss: 0.0311 - val_categorical_crossentropy: 0.0973 - val_tversky_coefficient: 0.8548 - val_dice_coefficient: 0.9035 - val_iou_coefficient: 0.8262
    Epoch 63/100
    630/630 [==============================] - 561s 891ms/step - loss: 0.0353 - categorical_crossentropy: 0.1052 - tversky_coefficient: 0.8519 - dice_coefficient: 0.9007 - iou_coefficient: 0.8221 - val_loss: 0.0359 - val_categorical_crossentropy: 0.1039 - val_tversky_coefficient: 0.8512 - val_dice_coefficient: 0.9007 - val_iou_coefficient: 0.8219
    Epoch 64/100
    630/630 [==============================] - 532s 844ms/step - loss: 0.0331 - categorical_crossentropy: 0.1002 - tversky_coefficient: 0.8562 - dice_coefficient: 0.9040 - iou_coefficient: 0.8273 - val_loss: 0.0556 - val_categorical_crossentropy: 0.1514 - val_tversky_coefficient: 0.8066 - val_dice_coefficient: 0.8665 - val_iou_coefficient: 0.7684
    Epoch 65/100
    630/630 [==============================] - 536s 850ms/step - loss: 0.0355 - categorical_crossentropy: 0.1049 - tversky_coefficient: 0.8524 - dice_coefficient: 0.9011 - iou_coefficient: 0.8226 - val_loss: 0.0285 - val_categorical_crossentropy: 0.1021 - val_tversky_coefficient: 0.8448 - val_dice_coefficient: 0.8961 - val_iou_coefficient: 0.8140
    Epoch 66/100
    630/630 [==============================] - 537s 852ms/step - loss: 0.0347 - categorical_crossentropy: 0.1039 - tversky_coefficient: 0.8530 - dice_coefficient: 0.9015 - iou_coefficient: 0.8233 - val_loss: 0.0317 - val_categorical_crossentropy: 0.0989 - val_tversky_coefficient: 0.8534 - val_dice_coefficient: 0.9024 - val_iou_coefficient: 0.8246
    Epoch 67/100
    630/630 [==============================] - 553s 877ms/step - loss: 0.0343 - categorical_crossentropy: 0.1026 - tversky_coefficient: 0.8543 - dice_coefficient: 0.9025 - iou_coefficient: 0.8250 - val_loss: 0.0726 - val_categorical_crossentropy: 0.1563 - val_tversky_coefficient: 0.8230 - val_dice_coefficient: 0.8789 - val_iou_coefficient: 0.7884
    Epoch 68/100
    630/630 [==============================] - 501s 795ms/step - loss: 0.0356 - categorical_crossentropy: 0.1058 - tversky_coefficient: 0.8513 - dice_coefficient: 0.9003 - iou_coefficient: 0.8213 - val_loss: 0.0321 - val_categorical_crossentropy: 0.0966 - val_tversky_coefficient: 0.8570 - val_dice_coefficient: 0.9051 - val_iou_coefficient: 0.8291



```python
eval_validation = ps.evaluate_models('examples/30_clouds_validation.csv')
eval_test = ps.evaluate_models('examples/30_clouds_test.csv')
```

    1680 images detected!
    Set 'steps_per_epoch' to: 53


    2022-10-19 12:40:59.976871: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8101


    53/53 [==============================] - 125s 2s/step - loss: 0.0351 - categorical_crossentropy: 0.1080 - dice_coefficient: 0.8957 - tversky_coefficient: 0.8435 - iou_coefficient: 0.8117
    1680 images detected!
    Set 'steps_per_epoch' to: 210
    210/210 [==============================] - 119s 560ms/step - loss: 0.0360 - dice_coefficient: 0.9117 - iou_coefficient: 0.8402 - categorical_crossentropy: 0.0940 - tversky_coefficient: 0.8660
    EVALUATION RESULTS:
    
                model_name  focal_loss  categorical_crossentropy  \
    0            38c_u_net    0.035124                  0.108038   
    0  38c_u_net_plus_plus    0.036038                  0.911740   
    
       dice_coefficient  iou_coefficient  tversky_coefficient  
    0          0.895686         0.843504             0.811663  
    0          0.840156         0.093998             0.865977  
    1680 images detected!
    Set 'steps_per_epoch' to: 53
    53/53 [==============================] - 133s 3s/step - loss: 0.0307 - categorical_crossentropy: 0.0972 - dice_coefficient: 0.9046 - tversky_coefficient: 0.8560 - iou_coefficient: 0.8262
    1680 images detected!
    Set 'steps_per_epoch' to: 210
    210/210 [==============================] - 118s 559ms/step - loss: 0.0313 - dice_coefficient: 0.9217 - iou_coefficient: 0.8569 - categorical_crossentropy: 0.0825 - tversky_coefficient: 0.8801
    EVALUATION RESULTS:
    
                model_name  focal_loss  categorical_crossentropy  \
    0            38c_u_net    0.030687                  0.097211   
    0  38c_u_net_plus_plus    0.031301                  0.921655   
    
       dice_coefficient  iou_coefficient  tversky_coefficient  
    0          0.904561         0.855966             0.826248  
    0          0.856911         0.082530             0.880135  


# Predictions

Only after do we train the models, we can easily produce predicted masks based on the validation set or whatever data that we would like to use, just make sure it is organized as in the train/validation/test sets.


```python
# When the custom_data_path is set to None, the validation data will be used.
# If that is not the intention of yours, feel free to point the engine to any other direction.pyplatypus.com
ps.produce_and_save_predicted_masks_for_model(model_name="38c_u_net", custom_data_path=None)
ps.produce_and_save_predicted_masks_for_model(model_name="38c_u_net", custom_data_path='examples/30_clouds_test.csv')
```

    1680 images detected!
    Set 'steps_per_epoch' to: 53
    1/1 [==============================] - 1s 537ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 43ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 41ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 42ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 41ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 41ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 43ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 41ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 42ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 41ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 41ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 42ms/step
    1/1 [==============================] - 0s 38ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 42ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 428ms/step
    1680 images detected!
    Set 'steps_per_epoch' to: 53
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 42ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 41ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 42ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 41ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 41ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 45ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 43ms/step
    1/1 [==============================] - 0s 42ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 32ms/step
    1/1 [==============================] - 0s 41ms/step
    1/1 [==============================] - 0s 42ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 42ms/step
    1/1 [==============================] - 0s 46ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 44ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 42ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 41ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 41ms/step
    1/1 [==============================] - 0s 25ms/step
    1/1 [==============================] - 0s 41ms/step
    1/1 [==============================] - 0s 42ms/step
    1/1 [==============================] - 0s 39ms/step
    1/1 [==============================] - 0s 25ms/step



```python
# When the custom_data_path is set to None, the validation data will be used.
# If that is not the intention of yours, feel free to point the engine to any other direction.pyplatypus.com
ps.produce_and_save_predicted_masks_for_model(model_name="38c_u_net_plus_plus", custom_data_path=None)
ps.produce_and_save_predicted_masks_for_model(model_name="38c_u_net_plus_plus", custom_data_path='examples/30_clouds_test.csv')
```

    1680 images detected!
    Set 'steps_per_epoch' to: 210
    1/1 [==============================] - 0s 361ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 25ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 25ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 26ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 26ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 26ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 25ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 26ms/step
    1/1 [==============================] - 0s 25ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 29ms/step
    1/1 [==============================] - 0s 25ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 36ms/step
    1/1 [==============================] - 0s 28ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 27ms/step
    1/1 [==============================] - 0s 27ms/step
    1/1 [==============================] - 0s 25ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 28ms/step
    1/1 [==============================] - 0s 26ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 32ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 27ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 31ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 27ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 27ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 25ms/step
    1/1 [==============================] - 0s 25ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 31ms/step
    1/1 [==============================] - 0s 25ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 27ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 27ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 18ms/step
    1/1 [==============================] - 0s 18ms/step
    1/1 [==============================] - 0s 17ms/step
    1/1 [==============================] - 0s 18ms/step
    1/1 [==============================] - 0s 18ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 18ms/step
    1/1 [==============================] - 0s 18ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 28ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 40ms/step
    1/1 [==============================] - 0s 26ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 25ms/step
    1/1 [==============================] - 0s 26ms/step
    1/1 [==============================] - 0s 25ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 18ms/step
    1/1 [==============================] - 0s 29ms/step
    1/1 [==============================] - 0s 17ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 18ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 18ms/step
    1/1 [==============================] - 0s 18ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1680 images detected!
    Set 'steps_per_epoch' to: 210
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 28ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 30ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 26ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 30ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 26ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 28ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 26ms/step
    1/1 [==============================] - 0s 28ms/step
    1/1 [==============================] - 0s 26ms/step
    1/1 [==============================] - 0s 25ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 26ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 27ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 26ms/step
    1/1 [==============================] - 0s 27ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 31ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 34ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 27ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 25ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 26ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 25ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 18ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 18ms/step
    1/1 [==============================] - 0s 18ms/step
    1/1 [==============================] - 0s 38ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 18ms/step
    1/1 [==============================] - 0s 18ms/step
    1/1 [==============================] - 0s 25ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 18ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 18ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 18ms/step
    1/1 [==============================] - 0s 18ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 25ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 24ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 23ms/step
    1/1 [==============================] - 0s 21ms/step

