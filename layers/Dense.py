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
