import numpy as np

# generate a fully connected layer


class Dense:
    def __init__(self, input_size, output_size):
        self.weights = 0.01 * np.random.randn(input_size, output_size)
        self.biases = np.zeros(output_size)
        return None

    def forward(self, inputs) -> np.ndarray:
        # calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
        # relu
        self.output = np.maximum(self.output, 0)
        return self.output

    def backward(self, grad_y_pred, learning_rate):
        # calculate gradients
        grad_weights = np.dot(self.inputs.T, grad_y_pred)
        grad_biases = np.sum(grad_y_pred, axis=0, keepdims=True)
        grad_inputs = np.dot(grad_y_pred, self.weights.T)
        # update parameters
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases
        return grad_inputs