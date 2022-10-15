import numpy as np
from itertools import product
from layers.layer import Layer

class Pooling(Layer):
    def __init__(self, spatial_extent:int=2, stride:int=2, mode:str='max'):
        """
        spatial_extent: the spatial extent of the pooling operation
        stride: the stride of the pooling operation
        mode: the mode of the pooling operation, either 'max' or 'average'
        """
        self.spatial_extent = spatial_extent
        self.stride = stride
        self.mode = mode
        super().__init__(f"Pooling {spatial_extent}x{spatial_extent} {stride} {mode}")
        return None

    def forward(self, X:np.ndarray) -> np.ndarray:
        N, h, w, d = X.shape
        h_out = int((h - self.spatial_extent) / self.stride) + 1
        w_out = int((w - self.spatial_extent) / self.stride) + 1
        d_out = d

        out = np.zeros((N, h_out, w_out, d_out))
        for i, j, k in product(range(h_out), range(w_out), range(d_out)):
            if self.mode == 'max':
                out[:, i, j, k] = np.max(X[:, i*self.stride:i*self.stride+self.spatial_extent, j*self.stride:j*self.stride+self.spatial_extent, k], axis=(1,2))
            elif self.mode == 'average':
                out[:, i, j, k] = np.mean(X[:, i*self.stride:i*self.stride+self.spatial_extent, j*self.stride:j*self.stride+self.spatial_extent, k], axis=(1,2))

        return out
    
    def backward(self, grad_y_pred:np.ndarray, learning_rate:float=0.01) -> np.ndarray:
        return None