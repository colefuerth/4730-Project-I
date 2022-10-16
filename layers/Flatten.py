import numpy as np
from layers.layer import Layer

class Flatten(Layer):
    def __init__(self):
        self.last_shape = None
        self.last_input = None
        super().__init__(f"Flatten")
        return None

    def forward(self, X:np.ndarray) -> np.ndarray:
        self.last_shape = X.shape
        self.last_input = X
        return X.reshape(X.shape[0], -1, X.shape[3])

    def backward(self, grad_y_pred:np.ndarray, learning_rate:float=0.01) -> np.ndarray:
        grad_y_pred = grad_y_pred.reshape(self.last_shape)
        return grad_y_pred