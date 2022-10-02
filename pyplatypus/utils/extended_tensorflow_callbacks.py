from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, BackupAndRestore,
    TerminateOnNaN, CSVLogger, ProgbarLogger
    )


class EarlyStoppingExtension(EarlyStopping):
    def __init__(self, input_dict):
        self.name = input_dict.get("name")
        input_dict.pop("name")
        super().__init__(**input_dict)


class ModelCheckpointExtension(ModelCheckpoint):
    def __init__(self, input_dict):
        self.name = input_dict.get("name")
        input_dict.pop("name")
        super().__init__(**input_dict)


class ReduceLROnPlateauExtension(ReduceLROnPlateau):
    def __init__(self, input_dict):
        self.name = input_dict.get("name")
        input_dict.pop("name")
        super().__init__(**input_dict)


class TensorBoardExtension(TensorBoard):
    def __init__(self, input_dict):
        self.name = input_dict.get("name")
        input_dict.pop("name")
        super().__init__(**input_dict)


class BackupAndRestoreExtension(BackupAndRestore):
    def __init__(self, input_dict):
        self.name = input_dict.get("name")
        input_dict.pop("name")
        super().__init__(**input_dict)


class TerminateOnNaNExtension(TerminateOnNaN):
    def __init__(self, input_dict):
        self.name = input_dict.get("name")
        input_dict.pop("name")
        super().__init__(**input_dict)


class CSVLoggerExtension(CSVLogger):
    def __init__(self, input_dict):
        self.name = input_dict.get("name")
        input_dict.pop("name")
        super().__init__(**input_dict)


class ProgbarLoggerExtension(ProgbarLogger):
    def __init__(self, input_dict):
        self.name = input_dict.get("name")
        input_dict.pop("name")
        super().__init__(**input_dict)
