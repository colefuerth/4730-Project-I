import numpy as np


class Conv2D:
    def __init__(self, num_filters, spatial_extent, stride, zero_padding):
        self.num_filters = num_filters
        self.spatial_extent = spatial_extent
        self.stride = stride
        self.zero_padding = zero_padding

        self.filters = self.filter_gen(spatial_extent, num_filters)

    def filter_gen(spatial_extent, num_filters):
        #filters = np.random.rand(num_filters, spatial_extent, spatial_extent) -0.5
        filters = np.random.randn(
            num_filters, spatial_extent, spatial_extent) / np.sqrt(spatial_extent * spatial_extent)
        return filters

    def forward(self, input):
