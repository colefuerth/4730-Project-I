import numpy as np

class Flatten:
    def __init__(self):
        return None

    def forward(self, X:np.ndarray) -> np.ndarray:
        return X.reshape(-1)

    def backward(self, grad_y_pred, learning_rate):
        return None