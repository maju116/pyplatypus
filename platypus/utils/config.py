from yaml import load, FullLoader


def load_config_from_yaml(
        config_path: str
) -> dict:
    """
    Loads configuration from YAML file.

    Returns:
        Configuration (dictionary).
    """
    with open(config_path) as cfg:
        config = load(cfg, Loader=FullLoader)
    return config

