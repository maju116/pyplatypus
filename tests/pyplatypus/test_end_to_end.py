from pyplatypus.solvers.platypus_cv_solver import PlatypusSolver
from pathlib import Path

def test_complete_run():
    pe = PlatypusSolver(config_yaml_path="tests/testdata/config_yaml/end_to_end_testing_config.yaml")
    pe.run()
    pe.produce_and_save_predicted_masks_for_model(model_name="res_u_net", custom_data_path=None)
    Path("tests/testdata/chkpt.hdf5").unlink()
    assert pe.cache.get("semantic_segmentation")
