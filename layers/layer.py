
class Layer:
    def __init__(self, name):
        self.name = name

    def forward(self, X):
        raise NotImplementedError
    
    def backward(self, grad_y_pred, learning_rate):
        raise NotImplementedError

    def __str__(self):
        return self.name
