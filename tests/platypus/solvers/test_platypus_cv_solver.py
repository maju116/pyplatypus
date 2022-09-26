from platypus.solvers.platypus_cv_solver import PlatypusSolver
import yaml
import pytest
from pathlib import Path


class TestPlatypusSolver:

    def test_init_inproper_path(self, tmpdir):
        with pytest.raises(NotADirectoryError):
            PlatypusSolver(config_yaml_path=Path(f"{tmpdir}/non_existent_path"))

    def test_init(self, tmpdir, mocker):
        path_for_mocked_config = Path(f"{tmpdir}/mocked_config.yaml")
        with open(path_for_mocked_config, "w") as stream:
            yaml.dump({}, stream)
        
        mocker.patch("platypus.utils.config_processing_functions.YamlConfigLoader.load", return_value={"Success": True})
        ps = PlatypusSolver(config_yaml_path=path_for_mocked_config)
        assert ps.config.get("Success")

    def test_run(self, mocker):
        mocker.patch("platypus.solvers.platypus_cv_solver.PlatypusSolver.__init__", return_value=None)
        mocker.patch("platypus.solvers.platypus_cv_solver.PlatypusSolver.train", return_value=None)
        assert PlatypusSolver(config_yaml_path=Path("")).run() is None
