from pyplatypus.solvers.platypus_cv_solver import PlatypusSolver

def test_complete_run():
    pe = PlatypusSolver(config_yaml_path="tests/testdata/config_yaml/end_to_end_testing_config.yaml")
    pe.run()
    assert pe.cache.get("semantic_segmentation")