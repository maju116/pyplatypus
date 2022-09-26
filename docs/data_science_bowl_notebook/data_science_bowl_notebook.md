# The data
This notebook aims at presenting the process of creating the end-to-end modelling pipeline with the use of [PyPlatypus](https://github.com/maju116/pyplatypus) package.

As starting point, the data is crucial, let's proceed with gathering it hence. We will be working with the nuclei-related images of which the [2018 Data Science Bowl dataset](https://www.kaggle.com/c/data-science-bowl-2018/data) is composed. These are PNG files organized in accordance with the following structure:

stage1_data

----|image_name

--------|images

------------|image.png

--------|masks

------------|mask1.png

------------|mask2.png

In the "images" folder you will find an image of shape (256, 256, 4) with the values varying from 0 to 1, while the masks are just (256, 256) matrices with the values coming from the discrete set: {0, 1}.

# The preparation
After downloading the data, unpack it and move to any preferred destination. For this example we will be interested only in stage1_train and stage1_test subdirectories, thus other files could be put aside. Let's take a look at the exemplary image.


```python
from pathlib import Path
import os
cwd = Path.cwd()
while cwd.stem != "pyplatypus":
    cwd = cwd.parent
os.chdir(cwd)
```


```python
# 20% for validation
data_path = Path("examples/data/data_science_bowl/")
models_path = Path("examples/models/")

example = "0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9"
train_image = os.listdir(data_path/Path(f"stage1_train/{example}/images"), )
train_masks = os.listdir(data_path/Path(f"stage1_train/{example}/masks"))
```


```python
from matplotlib import pyplot as plt
# Load the image along with on of the masks associated
example_image = plt.imread(data_path/Path(f"stage1_train/{example}/images/{train_image[0]}"))
example_mask1 = plt.imread(data_path/Path(f"stage1_train/{example}/masks/{train_masks[0]}"))
example_mask2 = plt.imread(data_path/Path(f"stage1_train/{example}/masks/{train_masks[2]}"))
```


```python
plt.figure()
plt.imshow(example_image, alpha=0.6)
plt.imshow(example_mask1, alpha=0.2)
plt.imshow(example_mask2, alpha=0.2)
plt.title("Original image with the masks in yellow.")
```




    Text(0.5, 1.0, 'Original image with the masks in yellow.')




    
![png](output_5_1.png)
    


What you see as the yellow area is the "Segmentation mask" which is simply telling us which pixel belongs to which class. Assesing this membership is the goal that we are striving to achieve.

# Splitting the data
For the modeling, beside train and test sets, we will need a validation set for it is common good practice to have one.
Feel free to use the following script but beware, it will move a sample set of images from yout stage1_train folder to the stage1_validation.


```python
from glob import glob
from random import sample
from shutil import move

def create_validation_set(data_path, p=0.2):
    train_images = glob(str(data_path/Path("stage1_train/*")))
    # Validation set size (as percentage)
    p = 0.2
    validation_images = sample(train_images, round(p*len(train_images)))
    Path.mkdir(data_path/Path("stage1_validation"))
    for img_path in validation_images:
        trimmed_path = Path(img_path).stem
        move(img_path, str(data_path/Path(f"stage1_validation/{trimmed_path}")))
```

Let's now inspect the input message that we are to send to PlatypusSolver in order to run it.


```python
import yaml
import json
with open(r"examples/data_science_bowl_config.yaml") as stream:
    config = yaml.safe_load(stream)
    print(json.dumps(config, indent=4, sort_keys=True))
```

    {
        "augmentation": {
            "Blur": {
                "always_apply": false,
                "blur_limit": 7,
                "p": 0.5
            },
            "Flip": {
                "always_apply": false,
                "p": 0.5
            },
            "RandomRotate90": {
                "always_apply": false,
                "p": 0.5
            },
            "ToFloat": {
                "always_apply": true,
                "max_value": 255,
                "p": 1.0
            }
        },
        "object_detection": null,
        "semantic_segmentation": {
            "data": {
                "colormap": [
                    [
                        0,
                        0,
                        0
                    ],
                    [
                        1,
                        1,
                        1
                    ]
                ],
                "column_sep": ";",
                "loss": "focal loss",
                "metrics": [
                    "tversky coefficient",
                    "iou coefficient"
                ],
                "mode": "nested_dirs",
                "optimizer": "adam",
                "shuffle": false,
                "subdirs": [
                    "images",
                    "masks"
                ],
                "test_path": "examples/data/data_science_bowl/stage1_test",
                "train_path": "examples/data/data_science_bowl/stage1_train",
                "validation_path": "examples/data/data_science_bowl/stage1_validation"
            },
            "models": [
                {
                    "activation_layer": "relu",
                    "batch_normalization": true,
                    "batch_size": 32,
                    "blocks": 4,
                    "deep_supervision": false,
                    "dropout": 0.2,
                    "epochs": 100,
                    "filters": 16,
                    "grayscale": false,
                    "h_splits": 0,
                    "kernel_initializer": "he_normal",
                    "linknet": false,
                    "n_class": 2,
                    "name": "res_u_net",
                    "net_h": 256,
                    "net_w": 256,
                    "plus_plus": false,
                    "resunet": true,
                    "u_net_conv_block_width": 4,
                    "use_separable_conv2d": true,
                    "use_spatial_droput2d": true,
                    "use_up_sampling2d": false,
                    "w_splits": 0
                },
                {
                    "batch_normalization": true,
                    "batch_size": 8,
                    "blocks": 2,
                    "deep_supervision": true,
                    "dropout": 0.2,
                    "epochs": 5,
                    "filters": 16,
                    "grayscale": false,
                    "h_splits": 0,
                    "kernel_initializer": "he_normal",
                    "linknet": false,
                    "n_class": 2,
                    "name": "u_net_plus_plus",
                    "net_h": 256,
                    "net_w": 256,
                    "plus_plus": true,
                    "use_separable_conv2d": true,
                    "use_spatial_dropout2d": true,
                    "use_up_sampling2d": true,
                    "w_splits": 0
                }
            ]
        }
    }


What might have struck you is that the config is organized so that it might potentially tell the Solver to train multiple models while using a complex augmentation pipeline and loss functions coming from the rather large set of ones available within the PyPlatypus framework.

![68747470733a2f2f69322e77702e636f6d2f6e657074756e652e61692f77702d636f6e74656e742f75706c6f6164732f552d6e65742d6172636869746563747572652e706e673f73736c3d31.webp](u_net.png)

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
from platypus.solvers.platypus_cv_solver import PlatypusSolver


ps = PlatypusSolver(
    config_yaml_path=Path("examples/data_science_bowl_config.yaml")
)
ps.train()
```

    WARNING:root:The current image 1f9e429c12f4477221b5b855a5f494fda2ef6d064ff75b061ffaf093e91758c5 is incomplete for it contains only masks or images!


    445 images detected!
    Set 'steps_per_epoch' to: 14
    112 images detected!
    Set 'steps_per_epoch' to: 4
    112 images detected!
    Set 'steps_per_epoch' to: 4
    14/14 [==============================] - 101s 7s/step - loss: 0.1199 - categorical_crossentropy: 0.5762 - tversky_coefficient: 0.4767 - iou_coefficient: 0.3130 - val_loss: 0.1058 - val_categorical_crossentropy: 0.5702 - val_tversky_coefficient: 0.4876 - val_iou_coefficient: 0.3225


    WARNING:root:The current image 1f9e429c12f4477221b5b855a5f494fda2ef6d064ff75b061ffaf093e91758c5 is incomplete for it contains only masks or images!


    445 images detected!
    Set 'steps_per_epoch' to: 56
    112 images detected!
    Set 'steps_per_epoch' to: 14
    112 images detected!
    Set 'steps_per_epoch' to: 14
    56/56 [==============================] - 90s 2s/step - loss: 0.0642 - iou_coefficient: 0.3874 - categorical_crossentropy: 0.4394 - tversky_coefficient: 0.5567 - val_loss: 0.0784 - val_iou_coefficient: 0.3544 - val_categorical_crossentropy: 0.5074 - val_tversky_coefficient: 0.5233


# Predictions

Only after do we train the models, we can easily produce predicted masks based on the validation set or whatever data that we would like to use, just make sure it is organized as in the train/validation/test sets.


```python
from glob import glob
from random import sample
from PIL import Image
import numpy as np

def sample_and_plot_predictions(data_path: Path, model_name: str, n=3):
    validation_images = glob(str(data_path/Path("stage1_validation/*")))
    # Sample size
    n_max = len(validation_images)
    n=n_max if n > n_max else n
    validation_images = sample(validation_images, n)
    for img_path in validation_images:
        img_name = img_path.split("/")[-1:][0]
        img = glob(f"{img_path}/images/*.png")[0]
        predictions = glob(f"{img_path}/predicted_masks/{model_name}_predicted_mask.png")[0]
        # Load images
        img_loaded = Image.open(img) 
        predictions_loaded = Image.open(predictions)
        original_size_scaled = (np.array(img_loaded.size)/2).astype(int)
        predictions_scaled = predictions_loaded.resize(original_size_scaled)
        # Plot image and predicted masks
        f, axarr = plt.subplots(1,2)
        plt.title(f"Image and predictions: {img_name}")
        axarr[0].imshow(img_loaded)
        axarr[1].imshow(predictions_scaled)
```


```python
# Clean the results of former runs
from glob import glob
from shutil import rmtree
masks = glob(str(data_path/"stage1_validation/**/predicted_*"))
for mask in masks:
    rmtree(mask)

```


```python
# When the custom_data_path is set to None, the validation data will be used.
# If that is not the intention of yours, feel free to point the engine to any other direction.
ps.produce_and_save_predicted_masks_for_model(model_name="u_net_plus_plus", custom_data_path=None)
```

    1/1 [==============================] - 1s 742ms/step
    1/1 [==============================] - 0s 442ms/step
    1/1 [==============================] - 0s 331ms/step
    1/1 [==============================] - 0s 355ms/step
    1/1 [==============================] - 0s 318ms/step
    1/1 [==============================] - 0s 310ms/step
    1/1 [==============================] - 0s 313ms/step
    1/1 [==============================] - 0s 430ms/step
    1/1 [==============================] - 0s 316ms/step
    1/1 [==============================] - 0s 309ms/step
    1/1 [==============================] - 0s 311ms/step
    1/1 [==============================] - 0s 325ms/step
    1/1 [==============================] - 0s 326ms/step
    1/1 [==============================] - 0s 295ms/step



```python
sample_and_plot_predictions(data_path, model_name="u_net_plus_plus", n=10)
```


    
![png](output_19_0.png)
    



    
![png](output_19_1.png)
    



    
![png](output_19_2.png)
    



    
![png](output_19_3.png)
    



    
![png](output_19_4.png)
    



    
![png](output_19_5.png)
    



    
![png](output_19_6.png)
    



    
![png](output_19_7.png)
    



    
![png](output_19_8.png)
    



    
![png](output_19_9.png)
    



```python
ps.produce_and_save_predicted_masks_for_model(model_name="res_u_net", custom_data_path=None)
```

    1/1 [==============================] - 2s 2s/step
    1/1 [==============================] - 1s 1s/step
    1/1 [==============================] - 1s 1s/step
    1/1 [==============================] - 1s 1s/step



```python
sample_and_plot_predictions(data_path, model_name="u_net_plus_plus", n=10)
```


    
![png](output_21_0.png)
    



    
![png](output_21_1.png)
    



    
![png](output_21_2.png)
    



    
![png](output_21_3.png)
    



    
![png](output_21_4.png)
    



    
![png](output_21_5.png)
    



    
![png](output_21_6.png)
    



    
![png](output_21_7.png)
    



    
![png](output_21_8.png)
    



    
![png](output_21_9.png)
    

