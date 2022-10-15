import numpy as np
from itertools import product
from layers.layer import Layer


class Conv2D(Layer):
    def __init__(self, num_filters:int, spatial_extent:int, stride:int, zero_padding:int):
        self.num_filters = num_filters
        self.spatial_extent = spatial_extent
        self.stride = stride
        self.zero_padding = zero_padding

        self.filters = np.random.randn(
            spatial_extent, spatial_extent, num_filters) / np.sqrt(spatial_extent * spatial_extent)
        #self.filters = np.random.rand(num_filters, spatial_extent, spatial_extent) -0.5
        super().__init__(f"Conv2D {num_filters} {spatial_extent}x{spatial_extent} {stride} {zero_padding}")

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
        X = np.pad(X, ((0, 0), (self.zero_padding, self.zero_padding),
                                (self.zero_padding, self.zero_padding), (0, 0)), 'constant')

        # expand the filters to be directly multiplied with the input
        filters = np.expand_dims(self.filters, axis=0)
        filters = np.repeat(filters, N, axis=0)
        # also expand X along the conv filter dimension
        X = np.expand_dims(X, axis=3)
        X = np.repeat(X, self.num_filters, axis=3)
        X = np.squeeze(X, axis=4)
        # loop over the output
        for i, j in product(range(output_height), range(output_width)):
            # calculate the start and end of the current "slice"
            start_i = i * self.stride
            end_i = start_i + self.spatial_extent
            start_j = j * self.stride
            end_j = start_j + self.spatial_extent

            # dot product of the filter and the image at each stride
            output[:, i, j, :] = np.sum(
                filters * X[:, start_i:end_i, start_j:end_j, :], axis=(1, 2))
                 
        return output


    def backward(self, grad_y_pred, learning_rate):
        return None
