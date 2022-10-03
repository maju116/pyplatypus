# The Input

The whole process relies upon the YAML config shaped by a user. We wanted to avoid the lack of flexibility that often comes with
the already-prepared frameworks. Therefore the said freedom of movement is guaranteed by the multitude of config fields and the configs
alongside the sub-configs being built in the form of the blocks defining each workflow.

## The high level YAML structure

See the main required fields below. Hereafter to these blocks, that each config is expected to be composed of, we will be referring using their names.

    object_detection    # Currently it is not utilized.
    semantic_segmentation   # Each task is defined by the data and models fields.
        data:    # Shapes the data pipeline, hence the generators.
            ...
        models:  # Steers the model-building process. It is expected to be a list. 
            ...
    augmentation:    # Defines the augmentation pipelines.
        ...

Below there is each field described more thoroughly.

### semantic_segmentation.data

| Name | Type | Allowed values | Exemplary value | Required |
|---|---|---|---|---|
| train_path | str | any | 'tests/testdata/nested_dirs/' |  yes  |
| validation_path | str | any | 'tests/testdata/nested_dirs/' |  yes  |
| test_path | str | any | 'tests/testdata/nested_dirs/' |  yes  |
| colormap | List[List]  | any | [[0, 0, 0], [255, 255, 255]] | yes |
| mode | str | 'nested_dirs' or 'config_file' | 'nested_dirs' | yes |
| shuffle | boolean | any | False | yes |
| subdirs | str | any | ['images', 'masks'] | yes |
| column_sep | str | any | ';' | yes |

### semantic_segmentation.models

It is defined as a list, with each element following the below structure:

| Name | Type           | Allowed values | Exemplary value                            | Required |
|---|----------------|---|--------------------------------------------|---|
| name | str            | any | 'res_u_net'                                |  yes  |
| fine_tuning_path | Optional[str]  | any | 'models/res_u_net.h5'                      |  yes  |
| net_h | positive int   | any | 300                                        |  yes  |
| net_w | positive int   | any | 300                                        |  yes  |
| h_splits | positive int   | any | 0                                          |  no, by default 0  |
| w_splits | positive int   | any | 0                                          |  no, by default 0  |
| grayscale | boolean        | any | False                                      |  no, by default False  |
| blocks | positive int   | any | 4                                          |  yes  |
| n_class | positive int   | any | 2                                          |  yes  |
| filters | positive int   | any | 16                                         |  yes  |
| dropout | positive float | any between 0 and 1 | 0.2                                        |  yes  |
| batch_normalization | boolean        | any | True                                       |  yes  |
| kernel_initializer | str            | any implemented in Tensorflow | 'he_normal'                                |  no, by default 'he_normal' |
| resunet | boolean        | any | True                                       |  no, by default False |
| linknet | boolean        | any | False                                      |  no, by default False |
| plus_plus | boolean        | any | False                                      |  no, by default False |
| deep_supervision | boolean        | any | False                                      |  yes  |
| use_separable_conv2d | boolean        | any | True                                       |  no, by default True |
| use_spatial_droput2d | boolean        | any | False                                      |  no, by default True |
| use_up_sampling2d | boolean        | any | False                                      |  no, by default False |
| u_net_conv_block_width | positive int   | any | 4                                          |  no, by default 2 |
| activation_layer | str            | any implemented in Tensorflow | 16                                         |  no, by default 'relu' |
| batch_size | positive int   | any | 32                                         |  no, by default 32  |
| epochs | positive int   | any | 100                                        |  no, by default 2  |
| loss | str            | One of the: ['iou_loss', 'focal_loss', 'dice_loss', 'cce_loss', 'cce_dice_loss', 'tversky_loss', 'focal_tversky_loss', 'combo_loss', 'lovasz_loss'] | 'lovasz_loss'                              | no, by default 'iou_loss' |
| metrics | List[str]      | Subset of the: ['iou_coefficient', 'tversky_coefficient', 'dice_coefficient'] | ['tversky coefficient', 'iou coefficient'] | no, by default ['iou_coefficient'] |
| optimizer | dict           | Described in the Optimizers section | Described in the Optimizers section        | no, by default Adam optimizer with default arguments |
| callbacks | list or dict   | Described in the Callbacks section | Described in the Callbacks section         | no, by default no callbacks are used |


### Optimizers

PyPlatypus allows us to use the optimizers of choice, any implemented in Tensorflow backend, to learn more about the algorithms and their arguments, visit
[tensorflow.org](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers).

The available optimizers are: ["Adadelta", "Adagrad", "Adam", "Adamax", "Ftrl", "Nadam", "RMSprop", "SGD"]. Below we described each possible optimizer setting.

#### model.optimizer

#### models.optimizer.Adadelta

| Name | Type | Allowed values | Exemplary value | Required |
|---|---|---|---|---|
| learning_rate | float | any | 0.5 |  no, by default 0.001  |
| rho | float | any | 0.95 |  no, by default 0.95  |
| epsilon | float | any | 1e-07 |  no, by default 1e-07  |

#### models.optimizer.Adagrad

| Name | Type | Allowed values | Exemplary value | Required |
|---|---|---|---|---|
| learning_rate | float | any | 0.5 |  no, by default 0.001  |
| initial_accumulator_value | float | any | 0.1 |  no, by default 0.1  |
| epsilon | float | any | 1e-07 |  no, by default 1e-07  |

#### models.optimizer.Adam

| Name | Type | Allowed values | Exemplary value | Required |
|---|---|---|---|---|
| learning_rate | float | any | 0.5 |  no, by default 0.001  |
| beta_1 | float | any | 0.9 |  no, by default 0.9  |
| beta_2 | float | any | 0.999 |  no, by default 0.999  |
| epsilon | float | any | 1e-07 |  no, by default 1e-07  |
| amsgrad | boolean | any | False |  no, by default False  |

#### models.optimizer.Adamax

| Name | Type | Allowed values | Exemplary value | Required |
|---|---|---|---|---|
| learning_rate | float | any | 0.5 |  no, by default 0.001  |
| beta_1 | float | any | 0.9 |  no, by default 0.9  |
| beta_2 | float | any | 0.999 |  no, by default 0.999  |
| epsilon | float | any | 1e-07 |  no, by default 1e-07  |

#### models.optimizer.Ftlr

| Name | Type | Allowed values | Exemplary value | Required |
|---|---|---|---|---|
| learning_rate | float | any | 0.5 |  no, by default 0.001  |
| learning_rate_power | float | any | -0.5 |  no, by default -0.5  |
| initial_accumulator_value | float | any | 0.1 |  no, by default 0.1  |
| l1_regularization_strength | float | any | 0.0 |  no, by default 0.0  |
| l2_regularization_strength | float | any | 0.0 |  no, by default 0.0  |
| l2_shrinkage_regularization_strength | float | any | 0.0 |  no, by default 0.0  |
| beta | float | any | 0.0 |  no, by default 0.0  |

#### models.optimizer.Nadam

| Name | Type | Allowed values | Exemplary value | Required |
|---|---|---|---|---|
| learning_rate | float | any | 0.5 |  no, by default 0.001  |
| beta_1 | float | any | 0.9 |  no, by default 0.9  |
| beta_2 | float | any | 0.999 |  no, by default 0.999  |
| epsilon | float | any | 1e-07 |  no, by default 1e-07  |

#### models.optimizer.RMSprop

| Name | Type | Allowed values | Exemplary value | Required |
|---|---|---|---|---|
| learning_rate | float | any | 0.5 |  no, by default 0.001  |
| rho | float | any | 0.9 |  no, by default 0.9  |
| momentum | float | any | 0.0 |  no, by default 0.0  |
| epsilon | float | any | 1e-07 |  no, by default 1e-07  |
| centered | boolean | any | False |  no, by default False |

#### models.optimizer.SGD

| Name | Type | Allowed values | Exemplary value | Required |
|---|---|---|---|---|
| learning_rate | float | any | 0.5 |  no, by default 0.01  |
| momentum | float | any | 0.0 |  no, by default 0.0  |
| nesterov | boolean | any | False |  no, by default False |




### Callbacks

PyPlatypus allows us to use the callbacks of choice, majority of the ones implemented in Tensorflow, to learn more about the methods and their arguments, visit
[tensorflow.org](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks).

available_callbacks = 
available_callbacks_without_specification = ["EarlyStopping", "ReduceLROnPlateau", "TerminateOnNaN", "ProgbarLogger"]

The available optimizers are: ["EarlyStopping", "ModelCheckpoint", "ReduceLROnPlateau", "TensorBoard", "BackupAndRestore", "TerminateOnNaN", "CSVLogger", "ProgbarLogger"].

Beware that if you specify the callback input field as a list, only the following callbacks are valid: ["EarlyStopping", "ReduceLROnPlateau", "TerminateOnNaN", "ProgbarLogger"].
It is due to the fact that the other callbacks require specifying a storage path, assigning default value to which we perceive as improper.

Below we described each possible callback setting.

#### model.callbacks

#### models.callbacks.EarlyStopping

| Name | Type | Allowed values | Exemplary value | Required |
|---|---|---|---|---|
| monitor | str | any | "val_loss" |  no, by default "val_loss" |
| min_delta | float | any | 0 |  no, by default 0 |
| patience | PositiveInt | any | 0 |  no, by default 0  |
| verbose | PositiveInt | one of {0, 1} | 0 |  no, by default 0  |
| mode | str | one of {"auto", "min", "max"} | "auto" |  no, by default "auto" |
| baseline | float or None | any | 0.1 |  no, by default None |
| restore_best_weights | bool | any | True |  no, by default False |

#### models.callbacks.ModelCheckpoint

| Name | Type | Allowed values | Exemplary value | Required |
|---|---|---|---|---|
| filepath | str | path to the to-be-created file in an existing directory | "existing_dir/{epoch:02d}_{val_loss:.2f}.hdf5" |  yes |
| monitor | str | any | "val_loss" |  no, by default "val_loss" |
| verbose | PositiveInt | one of {0, 1} | 0 |  no, by default 0  |
| save_best_only | bool | any | True |  no, by default False |
| save_weights_only | bool | any | True |  no, by default False |
| mode | str | one of {"auto", "min", "max"} | "auto" |  no, by default "auto" |
| save_freq | str or PositiveInt | positive integer or "epoch" | "epoch" |  no, by default "epoch" |
| initial_value_threshold | float or None | any | 0 |  no, by default None |

#### models.callbacks.ReduceLROnPlateauSpec

| Name | Type | Allowed values | Exemplary value | Required |
|---|---|---|---|---|
| monitor | str | any | "val_loss" |  no, by default "val_loss" |
| factor | float | any | 0 |  no, by default 0 |
| patience | PositiveInt | any | 0 |  no, by default 10  |
| verbose | PositiveInt | one of {0, 1} | 0 |  no, by default 0  |
| mode | str | one of {"auto", "min", "max"} | "auto" |  no, by default "auto" |
| min_delta | float | any | 0 |  no, by default 0.0001 |
| cooldown | PositiveInt | any | 0 |  no, by default 0  |
| min_lr | PositiveFloat | any | 0 |  no, by default 0  |


#### models.callbacks.TensorBoard

| Name | Type | Allowed values | Exemplary value | Required |
|---|---|---|---|---|
| log_dir | str | existing directory | "existing_dir" |  yes |
| histogram_freq | PositiveInt | any | 1 |  no, by default 0 |
| write_graph | bool | any | True |  no, by default True |
| write_images | bool | any | True |  no, by default False |
| write_steps_per_epoch | bool | any | True |  no, by default False |
| update_freq | str or PositiveInt | positive integer or one of {"epoch", "batch"} | "epoch" |  no, by default "epoch" |
| profile_batch | PositiveInt | any | 0 |  no, by default 0  |
| embeddings_freq | PositiveInt or tuple of PositiveInt | any | 0 |  no, by default 0  |
| embeddings_metadata | dict | any | refer to [tensorflow.org](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard) |  no, by default None  |

#### models.callbacks.BackupAndRestoreSpec

| Name | Type | Allowed values | Exemplary value | Required |
|---|---|---|---|---|
| backup_dir | str | existing directory | "existing_dir" |  yes |
| save_freq | str or PositiveInt | positive integer or "epoch" | "epoch" |  no, by default "epoch" |
| delete_checkpoint | bool | any | True |  no, by default True |

#### models.callbacks.TerminateOnNaN

It takes no parameters, should be placed as a sole name when using list of callbacks or as empty mapping in the Yaml, see the example config at the end of this page.

#### models.callbacks.CSVLogger

| Name | Type | Allowed values | Exemplary value | Required |
|---|---|---|---|---|
| filename | str | path to the to-be-created file in an existing directory | "existing_dir/{epoch:02d}_{val_loss:.2f}.hdf5" |  yes |
| separator | str | any | "," |  no, by default "," |
| append | bool | any | False |  no, by default False  |

#### models.callbacks.ProgbarLogger

| Name | Type | Allowed values | Exemplary value | Required |
|---|---|---|---|---|
| count_mode | str | one of {"steps", "samples"} | "samples" |  no, by default "samples" |
| stateful_metrics | tuple or list of str | any | ("val_loss") |  no, by default None |


### augmentation

Here just a few exemplary fields are presented, but you may put any of the Albumentations-powered transforms.
Find out more at the [albumentations.ai](https://albumentations.ai/docs/api_reference/augmentations/).

Some key points to keep in mind while building your own augmentation pipeline:

* The ordering of transforms defined in the config will be respected further in the engine itself.

* Not all of the Albumentations-supported transforms are available in the PyPlatypus. The ones that you may use in our package are listed here:
[
    'Blur', 'GaussianBlur', 'GlassBlur', 'MedianBlur', 'MotionBlur',
    'CLAHE', 'ChannelDropout', 'ChannelShuffle', 'ColorJitter', 'Downscale',
    'Emboss', 'Equalize', 'FancyPCA', 'GaussNoise', 'HueSaturationValue',
    'ISONoise', 'InvertImg', 'MultiplicativeNoise', 'Normalize', 'RGBShift',
    'RandomBrightnessContrast', 'RandomFog', 'RandomGamma', 'RandomRain',
    'RandomSnow', 'RandomShadow', 'RandomSunFlare', 'RandomToneCurve',
    'Sharpen', 'Solarize', 'Superpixels', 'ToSepia', 'Affine', 'CenterCrop',
    'CoarseDropout', 'Crop', 'CropAndPad', 'CropNonEmptyMaskIfExists',
    'ElasticTransform', 'Flip', 'GridDistortion', 'GridDropout', 'HorizontalFlip',
    'MaskDropout', 'OpticalDistortion', 'Perspective', 'PiecewiseAffine', 'RandomCrop',
    'RandomCropNearBBox', 'RandomGridShuffle', 'RandomResizedCrop', 'RandomRotate90',
    'RandomSizedBBoxSafeCrop', 'Rotate', 'SafeRotate', 'ShiftScaleRotate', 'Transpose',
    'VerticalFlip', 'FromFloat', 'ToFloat'
    ]

* Also beware the fact that the following transforms are the only ones available for the validation and test data, for the sake of good practices:
['FromFloat', 'ToFloat', 'InvertImg']

| Name | Type | Required |
|---|---|---|
| InvertImg | InvertImgSpec | no, initialized with the default attributes |
| Blur | BlurSpec | no, initialized with the default attributes  |
| Flip | FlipSpec |  no, initialized with the default attributes  |
| RandomRotate90 | RandomRotate90Spec | no, initialized with the default attributes |
| ToFloat | ToFloatSpec |  no, initialized with the default attributes |

#### augmentation.InvertImg

| Name | Type | Allowed values | Exemplary value | Required |
|---|---|---|---|---|
| always_apply | boolean | any | True |  no, by default True |
| p | positive float | any between 0 and 1 | 1 |  no, by default 1 |


#### augmentation.Blur

| Name | Type | Allowed values | Exemplary value | Required |
|---|---|---|---|---|
| blur_limit | positive float | any | 7 |  no, by default 7 |
| always_apply | boolean | any | True |  no, by default False |
| p | positive float | any between 0 and 1 | 1 |  no, by default 0.5 |


#### augmentation.Flip

| Name | Type | Allowed values | Exemplary value | Required |
|---|---|---|---|---|
| always_apply | boolean | any | False |  no, by default False |
| p | positive float | any between 0 and 1 | 0.5 |  no, by default 0.5 |

#### augmentation.RandomRotate90

| Name | Type | Allowed values | Exemplary value | Required |
|---|---|---|---|---|
| always_apply | boolean | any | False |  no, by default False |
| p | positive float | any between 0 and 1 | 0.5 |  no, by default 0.5 |

#### augmentation.ToFloat

| Name | Type | Allowed values | Exemplary value | Required |
|---|---|---|---|---|
| blur_limit | positive float | any | 255 |  no, by default 255 |
| always_apply | boolean | any | True |  no, by default True |
| p | positive float | any between 0 and 1 | 1 |  no, by default 1 |


## Example config
Having dived in the specifics we shall close this section with an example YAML config. Feel free to try it out!

    object_detection:
    semantic_segmentation:
        data:
            train_path: 'tests/testdata/nested_dirs/'
            validation_path: 'tests/testdata/nested_dirs/'
            test_path: 'tests/testdata/nested_dirs/'
            colormap: [[0, 0, 0], [255, 255, 255]]
            mode: 'nested_dirs'
            shuffle: False
            subdirs: ["images", "masks"]
            column_sep: ';'

        models:
            - name: 'res_u_net'
                net_h: 300
                net_w: 300
                h_splits: 0
                w_splits: 0
                grayscale: False
                blocks: 4
                n_class: 2
                filters: 16
                dropout: 0.2
                batch_normalization: True
                kernel_initializer: 'he_normal'
                resunet: True
                linknet: False
                plus_plus: False
                deep_supervision: False
                use_separable_conv2d: True
                use_spatial_droput2d: True
                use_up_sampling2d: False
                u_net_conv_block_width: 4
                activation_layer: "relu"
                batch_size: 32
                epochs: 100
                loss: 'focal loss'
                metrics: ['tversky coefficient', 'iou coefficient']
                optimizer:
                    Adam:
                        learning_rate: 0.001
                        beta_1: 0.9
                        beta_2: 0.999
                        epsilon: 1e-07
                        amsgrad: False
                callbacks:
                    EarlyStopping:
                        patience: 10
                    ProgbarLogger:
                        count_mode: "samples"
                    TerminateOnNaN:
            - name: 'u_net_plus_plus'
                net_h: 300
                net_w: 300
                h_splits: 0
                w_splits: 0
                grayscale: False
                blocks: 2
                n_class: 2
                filters: 16
                dropout: 0.2
                batch_normalization: True
                kernel_initializer: 'he_normal'
                linknet: False
                plus_plus: True
                deep_supervision: True
                use_separable_conv2d: True
                use_spatial_dropout2d: True
                use_up_sampling2d: True
                batch_size: 32
                epochs: 100
                loss: 'focal loss'
                metrics: ['tversky coefficient', 'iou coefficient']
                optimizer:
                    Adam:
                        learning_rate: 0.001
                        beta_1: 0.9
                        beta_2: 0.999
                        epsilon: 1e-07
                        amsgrad: False
    augmentation:
        InvertImg:
            always_apply: True
            p: 1
        Blur:
            blur_limit: 7
            always_apply: False
            p: 0.5
        Flip:
            always_apply: False
            p: 0.5
        RandomRotate90:
            always_apply: False
            p: 0.5
        ToFloat:
            max_value: 255
            always_apply: True
            p: 1.0









