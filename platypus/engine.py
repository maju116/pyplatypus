from platypus.utils.config_processing_functions import check_cv_tasks # TODO move to pydantic as well
from platypus.utils.augmentation import create_augmentation_pipeline
from platypus.segmentation.generator import segmentation_generator
from platypus.segmentation.models.u_net import u_net
import platypus.detection as det
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from platypus.data_models.platypus_engine_datamodel import PlatypusSolverInput


class platypus_engine:

    def __init__(
            self,
            config: PlatypusSolverInput
    ) -> None:
        """
        Performs Computer Vision tasks based on YAML config file.

        Args:
            config_yaml_path (str): Path to the config YAML file.
        """
        self.config = config

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
                augmentation_pipeline = create_augmentation_pipeline(self.config['augmentation'])
        else:
            augmentation_pipeline = None

        if 'semantic_segmentation' in cv_tasks_to_perform:
            for model_cfg in self.config['semantic_segmentation']['models']:
                train_data_generator = segmentation_generator(
                    path=self.config['semantic_segmentation']['data']['train_path'],
                    mode=self.config['semantic_segmentation']['data']['mode'],
                    colormap=self.config['semantic_segmentation']['data']['colormap'],
                    only_images=False,
                    net_h=model_cfg['net_h'],
                    net_w=model_cfg['net_w'],
                    grayscale=model_cfg['grayscale'],
                    augmentation_pipeline=augmentation_pipeline,
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
                    grayscale=model_cfg['grayscale'],
                    augmentation_pipeline=None,
                    batch_size=model_cfg['batch_size'],
                    shuffle=self.config['semantic_segmentation']['data']['shuffle'],
                    subdirs=self.config['semantic_segmentation']['data']['subdirs'],
                    column_sep=self.config['semantic_segmentation']['data']['column_sep']
                )
                # Ad function for model selection based on type!!!
                model = u_net(
                    net_h=model_cfg['net_h'],
                    net_w=model_cfg['net_w'],
                    grayscale=model_cfg['grayscale'],
                    blocks=model_cfg['blocks'],
                    n_class=model_cfg['n_class'],
                    filters=model_cfg['filters'],
                    dropout=model_cfg['dropout'],
                    batch_normalization=model_cfg['batch_normalization'],
                    kernel_initializer=model_cfg['kernel_initializer']
                )
                # Add options for selection!!!
                model.compile(
                    loss='binary_crossentropy',
                    optimizer='adam'
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
                        monitor='val_loss'
                    )]
                )
        return None
