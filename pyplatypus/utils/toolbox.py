"""This module offers the set of tools shared between the various tasks connected by the fact of being related to the Computer Vision topic.

Functions
---------
convert_to_snake_case(any_case: str)
    Converts any given string to the snake case.

convert_to_camel_case(any_case: str)
    Converts any given string to the camel case.
"""


def convert_to_snake_case(any_case: str) -> str:
    """
    Converts any given string to the snake case.

    Parameters
    ----------
    any_case: str
        String to convert.

    Returns
    -------
    snake_case: str
        Snake case string.

    Examples
    --------
    >>> convert_to_snake_case("Dice loss")
    'dice_loss'
    """
    snake_case = "_".join(any_case.lower().split(" "))
    return snake_case


def convert_to_camel_case(any_case: str) -> str:
    """
    Converts any given string to the snake case.

    Parameters
    ----------
    any_case: str
        String to convert.

    Returns
    -------
    camel_case: str
        Camel case string.

    Examples
    --------
    >>> convert_to_snake_case("Dice loss")
    'DiceLoss'
    """
    if "_" in any_case:
        splitted = any_case.split("_")
    elif " " in any_case:
        splitted = any_case.split(" ")
    else:
        raise ValueError(f"Unable to convert: {any_case} to CamelCase! No underscore or space detected!")
    splitted = [s.title() for s in splitted]
    camel_case = "".join(splitted)
    return camel_case
