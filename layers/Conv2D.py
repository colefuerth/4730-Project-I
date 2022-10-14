import numpy as np
from itertools import product


class Conv2D:
    def __init__(self, num_filters, spatial_extent, stride, zero_padding):
        self.num_filters = num_filters
        self.spatial_extent = spatial_extent
        self.stride = stride
        self.zero_padding = zero_padding

        self.filters = np.random.randn(
            num_filters, spatial_extent, spatial_extent) / np.sqrt(spatial_extent * spatial_extent)
        #self.filters = np.random.rand(num_filters, spatial_extent, spatial_extent) -0.5

    def forward(self, X):
        hight, width, depth = X.shape
        # add zero padding
        if self.zero_padding:
            X = np.pad(X, self.spatial_extent // 2)
        w_out = int((width - self.spatial_extent + (2*self.zero_padding)) / self.stride) + 1
        h_out = int((hight - self.spatial_extent + (2*self.zero_padding)) / self.stride) + 1
        d_out = self.num_filters

        Y = np.zeros((h_out, w_out, d_out))
        # dot product of the filter and the image at each stride
        for i, j, k in product(range(h_out), range(w_out), range(d_out)):
            # dot product of the filter and the image at each stride
            temp = X[i:i+self.spatial_extent, j:j +
                     self.spatial_extent, :] * self.filters[k, :, :]
            product = np.sum(temp)
            # weight (size of filter)
            weight = (self.spatial_extent * self.spatial_extent * depth) * self.num_filters
            # bais (number of filters)
            bais = self.num_filters
            # output
            Y[i, j] = weight-product+bais
        return Y
