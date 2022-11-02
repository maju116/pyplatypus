import pytest
from pathlib import Path
from pyplatypus.data_models.callbacks import (
    EarlyStoppingSpec, ModelCheckpointSpec, ReduceLROnPlateauSpec, TensorBoardSpec, BackupAndRestoreSpec, CSVLoggerSpec, ProgbarLoggerSpec
    )
from pydantic import ValidationError


class TestEarlyStoppingSpec:
    """The class is initialized with default parameters but each time one of them is
    replaced by improper value"""

    @pytest.mark.parametrize("class_update", [({"verbose": 3}), ({"mode": "invalid_mode"})])
    def test_validators(self, class_update):
        with pytest.raises(ValidationError):
            EarlyStoppingSpec(**class_update)


class TestModelCheckpointSpec:
    """The class is initialized with default parameters but each time one of them is
    replaced by improper value"""
   
    @pytest.mark.parametrize("class_update", [
        ({"verbose": 3}),
        ({"mode": "invalid_mode"}),
        ({"save_freq": "invalid_freq"})
        ])
    def test_validators(self, class_update, tmpdir):
        filepath = f"{tmpdir}/path"
        Path.mkdir(Path(f"{tmpdir}/path"))
        class_update.update({"filepath": filepath})
        with pytest.raises(ValidationError):
            ModelCheckpointSpec(**class_update)

    def test_filepath_validator(self):
        with pytest.raises(FileNotFoundError):
            ModelCheckpointSpec(filepath="non-existent-dir/non-existent-path")


class TestReduceLROnPlateauSpec:
    """The class is initialized with default parameters but each time one of them is
    replaced by improper value"""

    @pytest.mark.parametrize("class_update", [({"verbose": 3}), ({"mode": "invalid_mode"})])
    def test_validators(self, class_update):
        with pytest.raises(ValidationError):
            ReduceLROnPlateauSpec(**class_update)



class TestTensorBoardSpec:
    """The class is initialized with default parameters but each time one of them is
    replaced by improper value"""
   
    @pytest.mark.parametrize("class_update", [
        ({"update_freq": "invalid_freq"})
        ])
    def test_validators(self, class_update, tmpdir):
        filepath = f"{tmpdir}/path"
        Path.mkdir(Path(f"{tmpdir}/path"))
        class_update.update({"log_dir": filepath})
        with pytest.raises(ValidationError):
            TensorBoardSpec(**class_update)

    def test_filepath_validator(self):
        with pytest.raises(FileNotFoundError):
            TensorBoardSpec(log_dir="non-existent-dir")



class TestBackupAndRestoreSpec:
    """The class is initialized with default parameters but each time one of them is
    replaced by improper value"""
   
    @pytest.mark.parametrize("class_update", [
    ({"update_freq": "invalid_freq"})
    ])
    def test_validators(self, class_update, tmpdir):
        filepath = f"{tmpdir}/path"
        Path.mkdir(Path(f"{tmpdir}/path"))
        class_update.update({"log_dir": filepath})
        with pytest.raises(ValidationError):
            BackupAndRestoreSpec(**class_update)


    def test_filepath_validator(self):
        with pytest.raises(FileNotFoundError):
            BackupAndRestoreSpec(backup_dir="non-existent-dir")


class TestCSVLoggerSpec:
    """The class is initialized with default parameters but each time one of them is
    replaced by improper value"""
   
    def test_filepath_validator(self):
        with pytest.raises(FileNotFoundError):
            CSVLoggerSpec(filename="non-existent-dir/non-existent-file")



class TestProgbarLoggerSpec:
    """The class is initialized with default parameters but each time one of them is
    replaced by improper value"""

    @pytest.mark.parametrize("class_update", [({"count_mode": "invalid_mode"})])
    def test_validators(self, class_update):
        with pytest.raises(ValidationError):
            ProgbarLoggerSpec(**class_update)
