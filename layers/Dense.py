import numpy as np
from layer import layer

# generate a fully connected layer


class Dense(layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(
            input_size, output_size) / np.sqrt(input_size)
        self.biases = np.zeros((1, output_size))
        super().__init__(f"Dense {input_size} -> {output_size}")

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.inputs = X
        return np.dot(X, self.weights) + self.biases


    def backward(self, grad_y_pred: np.ndarray, learning_rate):
        grad_weights = np.dot(self.inputs.T, grad_y_pred)
        grad_biases = np.sum(grad_y_pred, axis=0, keepdims=True)
        grad_inputs = np.dot(grad_y_pred, self.weights.T)
        # update parameters
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases
        return grad_inputs
