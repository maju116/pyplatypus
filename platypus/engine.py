from platypus.utils.config_processing_functions import check_cv_tasks # TODO move to pydantic as well
from platypus.utils.augmentation import create_augmentation_pipeline
from platypus.segmentation.generator import segmentation_generator
from platypus.segmentation.loss import segmentation_loss
from platypus.segmentation.models.u_net import u_net
import platypus.detection as det
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from platypus.data_models.platypus_engine_datamodel import PlatypusSolverInput


class platypus_engine:

    def __init__(
            self,
            config: PlatypusSolverInput,
            cache: dict
    ) -> None:
        """
        Performs Computer Vision tasks based on the certain data model, defined via the Pydantic parser.

        Args:
            config: PlatypusSolverInput
                Stores the specific task-related configs nested as its attributes e.g. to the models that are to
                be trained can be accessed as config.semantic_segmentation.models.
            cache: dict
                Dictionary that could be used as the handy storage for both outputs and intermediate results.
        """
        self.config = dict(config)
        self.cache = cache

    def train(
        self
    ) -> None:
        """
        Creates the augmentation pipeline based on the input config. Then the function performs
        the selected tasks e.g. semantic segmentation, which consists of compilation and fitting the model
        using the train and validation data generators created prior to the fitting.
        """
        cv_tasks_to_perform = check_cv_tasks(self.config)  # Move to pydantic...
        if 'augmentation' in self.config.keys():
            if self.config['augmentation'] is not None:
                train_augmentation_pipeline = create_augmentation_pipeline(
                    augmentation_dict=dict(self.config['augmentation']),
                    train=True
                    )
                validation_augmentation_pipeline = create_augmentation_pipeline(dict(self.config['augmentation']), False)
        else:
            train_augmentation_pipeline = None
            validation_augmentation_pipeline = None

        if 'semantic_segmentation' in cv_tasks_to_perform:
            spec = self.config['semantic_segmentation']
            for model_cfg in self.config['semantic_segmentation'].models:
                train_data_generator = segmentation_generator(
                    path=spec.data.train_path,
                    mode=spec.data.mode,
                    colormap=spec.data.colormap,
                    only_images=False,
                    net_h=model_cfg.net_h,
                    net_w=model_cfg.net_w,
                    h_splits=model_cfg.h_splits,
                    w_splits=model_cfg.w_splits,
                    grayscale=model_cfg.grayscale,
                    augmentation_pipeline=train_augmentation_pipeline,
                    batch_size=model_cfg.batch_size,
                    shuffle=spec.data.shuffle,
                    subdirs=spec.data.subdirs,
                    column_sep=spec.data.column_sep
                )
                # Add only if selected!!!
                validation_data_generator = segmentation_generator(
                    path=spec.data.validation_path,
                    mode=spec.data.mode,
                    colormap=spec.data.colormap,
                    only_images=False,
                    net_h=model_cfg.net_h,
                    net_w=model_cfg.net_w,
                    h_splits=model_cfg.h_splits,
                    w_splits=model_cfg.w_splits,
                    grayscale=model_cfg.grayscale,
                    augmentation_pipeline=validation_augmentation_pipeline,
                    batch_size=model_cfg.batch_size,
                    shuffle=spec.data.shuffle,
                    subdirs=spec.data.subdirs,
                    column_sep=spec.data.column_sep
                )
                # Ad function for model selection based on type!!!
                model = u_net(
                    **dict(model_cfg)
                ).model
                # Add options for selection!!!
                sl = segmentation_loss(n_class=model_cfg.n_class, background_index=None)
                model.compile(
                    loss=sl.IoU_loss,
                    optimizer='adam',
                    metrics=['categorical_crossentropy', sl.dice_coefficient, sl.IoU_coefficient]
                )
                model.fit(
                    train_data_generator,
                    epochs=model_cfg.epochs,
                    steps_per_epoch=train_data_generator.steps_per_epoch,
                    validation_data=validation_data_generator,
                    validation_steps=validation_data_generator.steps_per_epoch,
                    callbacks=[ModelCheckpoint(
                        filepath=model_cfg.name + '.hdf5',
                        save_best_only=True,
                        monitor='val_IoU_coefficient',
                        mode='max'
                    ), EarlyStopping(
                        monitor='val_IoU_coefficient', mode='max', patience=5
                    )]
                )
        return None
