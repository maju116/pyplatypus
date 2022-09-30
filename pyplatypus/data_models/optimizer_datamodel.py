"""This script hold the optimizers' specification, for further information refer to: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers"""


from pydantic import BaseModel


class AdadeltaSpec(BaseModel):
    learning_rate: float = 0.001
    rho: float = 0.95
    epsilon: float = 1e-07
    name: str = "Adadelta"


class AdagradSpec(BaseModel):
    learning_rate: float = 0.001
    initial_accumulator_value: float = 0.1
    epsilon: float = 1e-07
    name: str = "Adagrad"


class AdamSpec(BaseModel):
    learning_rate: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-07
    amsgrad: bool = False
    name: str = 'Adam'


class AdamaxSpec(BaseModel):
    learning_rate: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-07
    name: str = 'Adamax'


class FtrlSpec(BaseModel):
    learning_rate: float = 0.001
    learning_rate_power: float = -0.5
    initial_accumulator_value: float = 0.1
    l1_regularization_strength: float = 0.0
    l2_regularization_strength: float = 0.0
    l2_shrinkage_regularization_strength: float = 0.0
    beta: float = 0.0
    name: str = 'Ftrl'


class NadamSpec(BaseModel):
    learning_rate: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-07
    name: str = 'Nadam'


class RMSpropSpec(BaseModel):
    learning_rate: float = 0.001
    rho: float = 0.9
    momentum: float = 0.0
    epsilon: float = 1e-07
    centered: bool = False
    name: str = 'RMSprop'


class SGDSpec(BaseModel):
    learning_rate: float = 0.01
    momentum: float = 0.0
    nesterov: bool = False
    name: str = 'SGD'
