import numpy as np
from numpy.random import randn

N, D_in, H, D_out = 64, 1000, 100, 10
x, y = randn(N, D_in), randn(N, D_out)
w1, w2 = randn(D_in, H), randn(H, D_out)

print("x.shape: ", x.shape)
print("y.shape: ", y.shape)

print("w1.shape: ", w1.shape)
print("w2.shape: ", w2.shape)

loss = 0
for t in range(2000):
    h = 1 / (1 + np.exp(-x.dot(w1)))
    y_pred = h.dot(w2)
    loss = np.square(y_pred - y).sum()
    #print(t, loss)

    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h.T.dot(grad_y_pred)
    grad_h = grad_y_pred.dot(w2.T)
    grad_w1 = x.T.dot(grad_h * h * (1 - h))
    
    w1 -= 1e-4 * grad_w1
    w2 -= 1e-4 * grad_w2

print("loss: ", loss)