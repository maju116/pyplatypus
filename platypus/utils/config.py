from yaml import load, FullLoader


def load_config_from_yaml(
        config_path: str
) -> dict:
    """
    Loads configuration from YAML file.

    Returns:
        Configuration (dict).
    """
    with open(config_path) as cfg:
        config = load(cfg, Loader=FullLoader)
    return config


def check_tasks(
        config: dict
) -> list:
    """
    Checks which Computer Vision tasks are to be performed.

    Args:
        config (dict): Configuration dictionary with tasks

    Returns:

    """
    available_tasks = ['object_detection', 'semantic_segmentation']
    return [task for task in config.keys() if config[task] is not None and task in available_tasks]
