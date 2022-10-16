import numpy as np
from itertools import product

from numpy.random import rand
from layers.layer import Layer


class Conv2D(Layer):
    def __init__(self, num_filters:int, spatial_extent:int, stride:int, zero_padding:int):
        self.num_filters = num_filters
        self.spatial_extent = spatial_extent
        self.stride = stride
        self.zero_padding = zero_padding

        self.filters = np.random.randn(num_filters, spatial_extent, spatial_extent) / np.sqrt(spatial_extent * spatial_extent)
        #self.filters = np.random.rand(num_filters, spatial_extent, spatial_extent) -0.5
        super().__init__(f"Conv2D {num_filters} {spatial_extent}x{spatial_extent} {stride} {zero_padding}")

    def forward(self, X):
        # X is a 4D array of shape (N, height, width, depth)
        # N is the number of images
        # height is the height of the image
        # width is the width of the image
        # depth is the number of channels
        N, height, width, depth = X.shape

        self.last_input = X

        # calculate the output shape
        output_height = int(
            (height - self.spatial_extent + 2 * self.zero_padding) / self.stride + 1)
        output_width = int(
            (width - self.spatial_extent + 2 * self.zero_padding) / self.stride + 1)

        # initialize the output
        output = np.zeros((N, output_height, output_width, depth, self.num_filters))

        # pad the input
        X = np.pad(X, ((0,0), (self.zero_padding, self.zero_padding),(self.zero_padding, self.zero_padding), (0,0)), 'constant')

        # loop over the output
        for k, i, j, d in product(range(N), range(output_height), range(output_width), range(depth)):
            # calculate the start and end of the current "slice"
            start_i = i * self.stride
            end_i = start_i + self.spatial_extent
            start_j = j * self.stride
            end_j = start_j + self.spatial_extent
            # dot product of the filter and the image at each stride
            output[k, i, j, d] = np.sum(self.filters * X[k, start_i:end_i, start_j:end_j, d])
                 
        return output


    def backward(self, grad_y_pred:np.ndarray, learning_rate:float=0.01):
        # X is a 4D array of shape (N, height, width, depth)
        N, height, width, depth = self.last_input.shape

        filter = np.zeros(self.filters.shape)

        for k, f in product(range(N), range(self.num_filters)):
            for i, j, d in product(range(height), range(width), range(depth)):
                start_i = i * self.stride
                end_i = start_i + self.spatial_extent
                start_j = j * self.stride
                end_j = start_j + self.spatial_extent
                filter[f, :, :] = self.filters[f] * self.last_input[k, start_i:end_i, start_j:end_j, d]

            #adjust the filters
            self.filters -= learning_rate * filter
        return None
    