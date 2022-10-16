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
        # N is the number of images
        # height is the height of the image
        # width is the width of the image
        # depth is the number of channels
        # filters is the number of filters
        N, height, width, depth, filters = X.shape

        self.last_input = X

        output_height = int((height - self.spatial_extent) / self.stride) + 1
        output_width = int((width - self.spatial_extent) / self.stride) + 1

        out = np.zeros((N, output_height, output_width, depth, filters))
        for k, i, j, d, f in product(range(N), range(output_height), range(output_width), range(depth), range(filters)):
            if self.mode == 'max':
                out[k, i, j, d, f] = np.max(X[k, i*self.stride:i*self.stride+self.spatial_extent, j*self.stride:j*self.stride+self.spatial_extent, d, f], axis=(0,1))
            elif self.mode == 'average':
                out[k, i, j, d] = np.mean(X[k, i*self.stride:i*self.stride+self.spatial_extent, j*self.stride:j*self.stride+self.spatial_extent, d], axis=(0,1))

        return out
    
    def backward(self, grad_y_pred:np.ndarray, learning_rate:float=0.01) -> np.ndarray:
        # N is the number of images
        # height is the height of the image
        # width is the width of the image
        # depth is the number of channels
        N, height, width, depth, filters = grad_y_pred.shape

        output_height = int((height - self.spatial_extent) / self.stride) + 1
        output_width = int((width - self.spatial_extent) / self.stride) + 1

        grad = np.zeros(self.last_input.shape)

        for k, i, j, d, f in product(range(N), range(output_height), range(output_width), range(depth), range(filters) ):
            grad_max = np.max(self.last_input[k, i*self.stride:i*self.stride+self.spatial_extent, j*self.stride:j*self.stride+self.spatial_extent, d, f], axis=(0,1))
            for i2, j2 in product(range(self.spatial_extent), range(self.spatial_extent)):
                if self.last_input[k, i*self.stride+i2, j*self.stride+j2, d, f] == grad_max:
                    grad[k, i*self.stride+i2, j*self.stride+j2, d, f] = grad_y_pred[k, i, j, d, f]
                    break
        return grad