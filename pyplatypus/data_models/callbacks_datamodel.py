"""This script hold the callbacks' specification, for further information refer to:
https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/"""

import pydantic
from pydantic import BaseModel
from pydantic import PositiveInt, PositiveFloat
from typing import Optional, Union, Tuple, List
from pathlib import Path


class EarlyStoppingSpec(BaseModel):
    monitor: str = "val_loss"
    min_delta: float = 0
    patience: PositiveInt = 0
    verbose: PositiveInt = 0
    mode: str = "auto"
    baseline: Optional[float] = None
    restore_best_weights: bool = False
    name: str = "EarlyStopping"

    @pydantic.validator("verbose", allow_reuse=True)
    def check_verbose(cls, v):
        if v in [0, 1]:
            return v
        raise ValueError(f"The chosen verbosity mode: {v} is not one of: [0, 1].")

    @pydantic.validator("mode", allow_reuse=True)
    def check_mode(cls, v):
        if v in ["auto", "min", "max"]:
            return v
        raise ValueError(f"The chosen mode: {v} is not one of [auto, min, max].")


class ModelCheckpointSpec(BaseModel):
    filepath: str
    monitor: str = "val_loss"
    verbose: PositiveInt = 0
    save_best_only: bool = False
    save_weights_only: bool = False
    mode: str = "auto"
    save_freq: Union[str, PositiveInt] = "epoch"
    initial_value_threshold: float = None
    name: str = "ModelCheckpoint"

    @pydantic.validator("filepath")
    def check_filepath(cls, v):
        if Path(v).parent.exists():
            return v
        raise FileNotFoundError(f"The chosen directory: {v} does not exist.")

    @pydantic.validator("verbose", allow_reuse=True)
    def check_verbose(cls, v):
        if v in [0, 1]:
            return v
        raise ValueError(f"The chosen verbosity mode: {v} is not one of: [0, 1].")

    @pydantic.validator("mode", allow_reuse=True)
    def check_mode(cls, v):
        if v in ["auto", "min", "max"]:
            return v
        raise ValueError(f"The chosen mode: {v} is not one of [auto, min, max].")

    @pydantic.validator("save_freq", allow_reuse=True)
    def check_save_freq(cls, v):
        if not isinstance(v, int):
            if v in ["epoch"]:
                return v
            raise ValueError("The save_freq may be integer or 'epoch'.")


class ReduceLROnPlateauSpec(BaseModel):
    monitor: str = "val_loss"
    factor: float = 0.1
    patience: PositiveInt = 10
    verbose: PositiveInt = 0
    mode: str = "auto"
    min_delta: float = 0.0001
    cooldown: PositiveInt = 0
    min_lr: PositiveFloat = 0
    name: str = "ReduceLROnPlateau"

    @pydantic.validator("verbose", allow_reuse=True)
    def check_verbose(cls, v):
        if v in [0, 1]:
            return v
        raise ValueError(f"The chosen verbosity mode: {v} is not one of: [0, 1].")

    @pydantic.validator("mode", allow_reuse=True)
    def check_mode(cls, v):
        if v in ["auto", "min", "max"]:
            return v
        raise ValueError(f"The chosen mode: {v} is not one of [auto, min, max].")


class TensorBoardSpec(BaseModel):
    log_dir: str
    histogram_freq: PositiveInt = 0
    write_graph: bool = True
    write_images: bool = False
    write_steps_per_epoch: bool = False
    update_freq: Union[str, PositiveInt] = "epoch"
    profile_batch: PositiveInt = 0
    embeddings_freq: Union[PositiveInt, Tuple[PositiveInt]] = 0
    embeddings_metadata: dict = None
    name: str = "TensorBoard"

    @pydantic.validator("log_dir")
    def check_logdir(cls, v):
        if Path(v).exists():
            return v
        raise FileNotFoundError(f"The chosen directory: {v} doeas not exist.")

    @pydantic.validator("update_freq")
    def check_update_freq(cls, v):
        if not isinstance(v, int):
            if v in ["epoch", "batch"]:
                return v
            raise ValueError("The save_freq may be integer or 'epoch' or 'batch'.")


class BackupAndRestoreSpec(BaseModel):
    backup_dir: str
    save_freq: Union[str, PositiveInt] = "epoch"
    delete_checkpoint: bool = True
    name: str = "BackupAndRestore"

    @pydantic.validator("backup_dir")
    def check_backupdir(cls, v):
        if Path(v).exists():
            return v
        raise FileNotFoundError(f"The chosen directory: {v} doeas not exist.")

    @pydantic.validator("save_freq")
    def check_save_freq(cls, v):
        if not isinstance(v, int):
            if v in ["epoch"]:
                return v
            raise ValueError("The save_freq may be integer or 'epoch'.")


class TerminateOnNaNSpec(BaseModel):
    name: str = "TerminateOnNaN"


class CSVLoggerSpec(BaseModel):
    filename: str
    separator: str = ','
    append: bool = False
    name: str = "CSVLogger"

    @pydantic.validator("filename")
    def check_filename(cls, v):
        if Path(v).parent.exists():
            return v
        raise FileNotFoundError(f"The chosen directory: {v} does not exist.")


class ProgbarLoggerSpec(BaseModel):
    count_mode: str = "samples"
    stateful_metrics: Union[Tuple[str], List[str]] = None
    name: str = "ProgbarLogger"

    @pydantic.validator("count_mode")
    def check_count_mode(cls, v):
        if v in ["steps", "samples"]:
            return v
        raise ValueError(f"The selected count mode: {v} is not one of [steps, samples].")
