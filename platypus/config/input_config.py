implemented_modes = ["nested_dirs", "config_file"]
implemented_losses = [
    "iou_loss", "focal_loss", "dice_loss", "cce_loss", "cce_dice_loss",
    "tversky_loss", "focal_tversky_loss", "combo_loss", "lovasz_loss"
    ]
implemented_metrics = ["iou_coefficient", "tversky_coefficient", "dice_coefficient"]
implemented_optimizers = [
    "adam", "adadelta", "adagrad", "adamax", "ftrl", "nadam", "rmsprop", "sgd"
    ]
available_activations = ["relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu", "exponential"]
