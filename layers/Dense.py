import numpy as np
from layers.layer import Layer
from itertools import product

# generate a fully connected layer


class Dense(Layer):
    def __init__(self, input_size, nodes):
        #self.weights = np.random.randn(input_size, nodes) / np.sqrt(input_size)
        self.nodes = nodes
        self.weights = np.random.randn(input_size, nodes) / input_size
        self.biases = np.zeros((1, nodes))
        super().__init__(f"Dense {input_size} -> {nodes}")


    def forward(self, X: np.ndarray) -> np.ndarray:
        # N is the number of images
        # height is the height of the image
        # width is the width of the image
        # depth is the number of channels
        # filters is the number of filters
        N, images, depth = X.shape

        self.last_input = X

        total = np.zeros((N, self. nodes, depth))
        for k, d in product(range(N), range(depth)):
            exp = np.exp(np.dot(X[k, :, d], self.weights) + self.biases)
            total[k, :, d] = exp / np.sum(exp)

        self.last_total = total

        return total


    def backward(self, grad_y_pred: np.ndarray, learning_rate:float = 1e-4):
        out = np.zeros(self.last_input.shape)
        for k , d in product(range(grad_y_pred.shape[0]), range(grad_y_pred.shape[2])):
            for i, grad in enumerate(grad_y_pred[k]):
                if (grad[d] == 0):
                    continue

                exp = np.exp(self.last_total[k, :, d])
                S = np.sum(exp)

                d_out_d_t = -exp[i] * exp / (S ** 2)
                d_out_d_t[i] = exp[i] * (S - exp[i]) / (S ** 2)

                d_t_d_w = self.last_input[k, :, d]
                d_t_d_b = 1
                d_t_d_inputs = self.weights

                d_l_d_t = grad[d] * d_out_d_t

                d_l_d_w = d_t_d_w[np.newaxis].T @ d_l_d_t[np.newaxis]
                d_l_d_b = d_l_d_t * d_t_d_b
                d_l_d_inputs = d_t_d_inputs @ d_l_d_t

                self.weights -= learning_rate * d_l_d_w
                self.biases -= learning_rate * d_l_d_b

                out[k, :, d] = d_l_d_inputs
        return out