import numpy as np
class Layer:
    def __init__(self, name, activation=None):
        self.name = name
        self._activation = activation
        if self._activation not in [None, 'relu', 'sigmoid', 'softmax']:
            raise Exception('Invalid activation function')

    def forward(self, X):
        raise NotImplementedError
    
    def backward(self, grad_y_pred, learning_rate):
        raise NotImplementedError

    def __str__(self):
        return self.name

    
    def activation(self, X):
        if not self._activation:
            return X

        if self._activation == 'relu':
            return np.maximum(X, 0)

        if self._activation == 'softmax':
            exps = np.exp(X - np.max(X))
            return exps / np.sum(exps, axis=1, keepdims=True)

        if self._activation == 'sigmoid':
            return 1 / (1 + np.exp(-X))
        
        raise Exception('Invalid activation function')
