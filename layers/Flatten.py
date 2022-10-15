import numpy as np

class Flatten:
    def __init__(self):
        return None

    def forward(self, X:np.ndarray) -> np.ndarray:
        N, h, w, d = X.shape
        return X.reshape((N, h*w*d))

    def backward(self, grad_y_pred:np.ndarray, learning_rate:float=0.01) -> np.ndarray:
        return None