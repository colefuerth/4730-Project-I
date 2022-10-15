import numpy as np
from layers.layer import Layer

# generate a fully connected layer


class Dense(Layer):
    def __init__(self, input_size, output_size, activation=None):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.biases = np.zeros((1, output_size))
        super().__init__(f"Dense {input_size} -> {output_size}", activation)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.inputs = X
        X = np.dot(X, self.weights) + self.biases
        return self.activation(X)

    def backward(self, grad_y_pred: np.ndarray, learning_rate: float = 1e-4):
        grad_w = np.dot(grad_y_pred, self.weights.T)
        grad_b = np.sum(grad_y_pred, axis=0, keepdims=True)
        self.weights -= learning_rate * np.dot(self.inputs.T, grad_y_pred)
        self.biases -= learning_rate * grad_b

        return grad_w

