import numpy as np
from itertools import product
from layers.layer import Layer


class Conv2D(Layer):
    def __init__(self, num_filters: int, spatial_extent: int, stride: int, zero_padding: int, activation: str = None):
        self.num_filters = num_filters
        self.spatial_extent = spatial_extent
        self.stride = stride
        self.zero_padding = zero_padding

        self.filters = np.random.randn(
            spatial_extent, spatial_extent, num_filters) / np.sqrt(spatial_extent)
        super().__init__(
            f"Conv2D {num_filters} {spatial_extent}x{spatial_extent} {stride} {zero_padding}", activation)

    def forward(self, X):
        # X is a 4D shape of N x H x W x C
        # N is the number of images
        # H is the height of the image
        # W is the width of the image
        # C is the number of channels
        N, height, width, depth = X.shape

        self.last_input = X

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

        return self.activation(output)

    def iterate_regions(self, image):
        # generates all possible 3*3 image regions using valid padding

        h, w = image.shape

        for i in range(h - self.spatial_extent + 1):
            for j in range(w - self.spatial_extent + 1):
                im_region = image[i:i + self.spatial_extent,
                                  j:j + self.spatial_extent]
                yield im_region, i, j

    def backward(self, grad_y_pred, learn_rate):

        return None  # dont do this for now
        # X is a 4D array of shape (N, height, width, depth)
        N, height, width, depth = grad_y_pred.shape
        print("grad_y_pred", grad_y_pred.shape)

        '''
        Performs a backward pass of the conv layer.
        - d_L_d_out is the loss gradient for this layer's outputs.
        - learn_rate is a float.
        '''
        for im_num in range(N):
            d_l_d_filters = np.zeros(self.filters.shape)
            for im_region, i, j in self.iterate_regions(self.last_input[im_num, :, :, 0]):
                for f in range(self.num_filters):
                    d_l_d_filters[:, :, f] += grad_y_pred[im_num,
                                                          i, j, f] * im_region

            # update filters
            self.filters -= learn_rate * d_l_d_filters

        return None
