import numpy as np
from itertools import product


class Conv2D:
    def __init__(self, num_filters:int, spatial_extent:int, stride:int, zero_padding:int):
        self.num_filters = num_filters
        self.spatial_extent = spatial_extent
        self.stride = stride
        self.zero_padding = zero_padding

        self.filters = np.random.randn(
            num_filters, spatial_extent, spatial_extent) / np.sqrt(spatial_extent * spatial_extent)
        #self.filters = np.random.rand(num_filters, spatial_extent, spatial_extent) -0.5

    def forward(self, X):
        # X is a 4D shape of N x H x W x C
        # N is the number of images
        # H is the height of the image
        # W is the width of the image
        # C is the number of channels
        N, height, width, depth = X.shape

        # calculate the output shape
        output_height = int(
            (height - self.spatial_extent + 2 * self.zero_padding) / self.stride + 1)
        output_width = int(
            (width - self.spatial_extent + 2 * self.zero_padding) / self.stride + 1)

        # initialize the output
        output = np.zeros((N, output_height, output_width, self.num_filters))

        # pad the input
        X_padded = np.pad(X, ((0, 0), (self.zero_padding, self.zero_padding),
                                (self.zero_padding, self.zero_padding), (0, 0)), 'constant')

        # loop over the output
        for i, j, k in product(range(output_height), range(output_width), range(self.num_filters)):
            # calculate the start and end of the current "slice"
            start_i = i * self.stride
            end_i = start_i + self.spatial_extent
            start_j = j * self.stride
            end_j = start_j + self.spatial_extent

            # slice the input and perform the convolution operation
            output[:, i, j, k] = np.sum(X_padded[:, start_i:end_i, start_j:end_j, :] * self.filters[k, :, :, :], axis=(1, 2, 3))

        return output


    def backward(self, grad_y_pred, learning_rate):
        return None