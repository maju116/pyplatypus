from platypus.utils.config import load_config_from_yaml, check_cv_tasks
from platypus.utils.augmentation import create_augmentation_pipeline
from platypus.segmentation.generator import segmentation_generator
from platypus.segmentation.loss import segmentation_loss
from platypus.segmentation.models.u_net import u_net
from platypus.segmentation.models.u_net_plus_plus import u_net_plus_plus
import platypus.detection as det
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class platypus_engine:

    def __init__(
            self,
            config_yaml_path: str
    ) -> None:
        """
        Performs Computer Vision tasks based on YAML config file.

        Args:
            config_yaml_path (str): Path to the config YAML file.
        """
        self.config = load_config_from_yaml(config_path=config_yaml_path)

    def train(
            self
    ) -> None:
        """
        Trains selected CV models.

        Returns:

        """
        cv_tasks_to_perform = check_cv_tasks(self.config)
        if 'augmentation' in self.config.keys():
            if self.config['augmentation'] is not None:
                train_augmentation_pipeline = create_augmentation_pipeline(self.config['augmentation'], True)
                validation_augmentation_pipeline = create_augmentation_pipeline(self.config['augmentation'], False)
        else:
            train_augmentation_pipeline = None
            validation_augmentation_pipeline = None

        if 'semantic_segmentation' in cv_tasks_to_perform:
            for model_cfg in self.config['semantic_segmentation']['models']:
                train_data_generator = segmentation_generator(
                    path=self.config['semantic_segmentation']['data']['train_path'],
                    mode=self.config['semantic_segmentation']['data']['mode'],
                    colormap=self.config['semantic_segmentation']['data']['colormap'],
                    only_images=False,
                    net_h=model_cfg['net_h'],
                    net_w=model_cfg['net_w'],
                    h_splits=model_cfg['h_splits'],
                    w_splits=model_cfg['w_splits'],
                    grayscale=model_cfg['grayscale'],
                    augmentation_pipeline=train_augmentation_pipeline,
                    batch_size=model_cfg['batch_size'],
                    shuffle=self.config['semantic_segmentation']['data']['shuffle'],
                    subdirs=self.config['semantic_segmentation']['data']['subdirs'],
                    column_sep=self.config['semantic_segmentation']['data']['column_sep']
                )
                # Add only if selected!!!
                validation_data_generator = segmentation_generator(
                    path=self.config['semantic_segmentation']['data']['validation_path'],
                    mode=self.config['semantic_segmentation']['data']['mode'],
                    colormap=self.config['semantic_segmentation']['data']['colormap'],
                    only_images=False,
                    net_h=model_cfg['net_h'],
                    net_w=model_cfg['net_w'],
                    h_splits=model_cfg['h_splits'],
                    w_splits=model_cfg['w_splits'],
                    grayscale=model_cfg['grayscale'],
                    augmentation_pipeline=validation_augmentation_pipeline,
                    batch_size=model_cfg['batch_size'],
                    shuffle=self.config['semantic_segmentation']['data']['shuffle'],
                    subdirs=self.config['semantic_segmentation']['data']['subdirs'],
                    column_sep=self.config['semantic_segmentation']['data']['column_sep']
                )
                # Ad function for model selection based on type!!!
                model = u_net_plus_plus(
                    net_h=model_cfg['net_h'],
                    net_w=model_cfg['net_w'],
                    grayscale=model_cfg['grayscale'],
                    blocks=model_cfg['blocks'],
                    n_class=model_cfg['n_class'],
                    filters=model_cfg['filters'],
                    dropout=model_cfg['dropout'],
                    batch_normalization=model_cfg['batch_normalization'],
                    kernel_initializer=model_cfg['kernel_initializer']
                ).model
                # Add options for selection!!!
                sl = segmentation_loss(n_class=model_cfg['n_class'], background_index=None)
                model.compile(
                    loss=sl.IoU_loss,
                    optimizer='adam',
                    metrics=['categorical_crossentropy', sl.dice_coefficient, sl.IoU_coefficient]
                )
                model.fit(
                    train_data_generator,
                    epochs=model_cfg['epochs'],
                    steps_per_epoch=train_data_generator.steps_per_epoch,
                    validation_data=validation_data_generator,
                    validation_steps=validation_data_generator.steps_per_epoch,
                    callbacks=[ModelCheckpoint(
                        filepath=model_cfg['name'] + '.hdf5',
                        save_best_only=True,
                        monitor='val_IoU_coefficient',
                        mode='max'
                    ), EarlyStopping(
                        monitor='val_IoU_coefficient', mode='max', patience=5
                    )]
                )
        return None
