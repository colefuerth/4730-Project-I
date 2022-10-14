import numpy as np

# generate a fully connected layer


class Dense:
    def __init__(self, input_size, output_size):
        self.weights = 0.01 * np.randn(input_size, output_size)
        self.biases = np.zeros(output_size)
        return None

    def run(self, inputs) -> np.ndarray:
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
