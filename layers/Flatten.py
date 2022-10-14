import numpy as np

class Flatten:
    def __init__(self):
        return None

    def forward(self, X):
        return X.reshape(X.shape[0], -1)

    def backward(self, grad_y_pred, learning_rate):
        return None