def convert_to_snake_case(any_case: str):
    """
    Converts the name to the snake case.

    Parameters
    ----------
    any_case: str
        Name to convert e.g. "Dice loss"

    Returns
    -------
    snake_case: str
        Snake case string e.g. "dice_loss"
    """
    snake_case = "_".join(any_case.lower().split(" "))
    return snake_case
