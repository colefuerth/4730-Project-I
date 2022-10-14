import numpy as np
from itertools import product

class Pooling:
    def __init__(self, spatial_extent=2, stride=2, mode='max'):
        self.spatial_extent = spatial_extent
        self.stride = stride
        self.mode = mode

    def run(self, input_data):
        X = np.asarray(X)
        h, w, d = X.shape
        h_out = int((h - self.spatial_extent) / self.stride) + 1
        w_out = int((w - self.spatial_extent) / self.stride) + 1
        d_out = d

        out = np.zeros((h_out, w_out, d_out))
        for i, j, k in product(range(h_out), range(w_out)):
            if self.mode == 'max':
                out[i, j, k] = np.max(X[i*self.stride:i*self.stride+self.spatial_extent, j*self.stride:j*self.stride+self.spatial_extent, k])
            elif self.mode == 'average':
                out[i, j, k] = np.mean(X[i*self.stride:i*self.stride+self.spatial_extent, j*self.stride:j*self.stride+self.spatial_extent, k])

        return out