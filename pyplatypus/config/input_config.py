"""This scripts is comprised of the various implemented scenarios for different fields in the user-defined config."""

implemented_modes = ["nested_dirs", "config_file"]
implemented_losses = [
    "iou_loss", "focal_loss", "dice_loss", "cce_loss", "cce_dice_loss",
    "tversky_loss", "focal_tversky_loss", "combo_loss", "lovasz_loss"
    ]
implemented_metrics = ["iou_coefficient", "tversky_coefficient", "dice_coefficient"]
available_activations = ["relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu", "exponential"]
available_optimizers = ["Adadelta", "Adagrad", "Adam", "Adamax", "Ftrl", "Nadam", "RMSprop", "SGD"]
available_callbacks = [
    "EarlyStopping", "ModelCheckpoint", "ReduceLROnPlateau", "TensorBoard", "BackupAndRestore", "TerminateOnNaN",
    "CSVLogger", "ProgbarLogger"
    ]
available_callbacks_without_specification = ["EarlyStopping", "ReduceLROnPlateau", "TerminateOnNaN", "ProgbarLogger"]
