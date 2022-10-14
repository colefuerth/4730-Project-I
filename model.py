# %%
# load the dataset

from tensorflow.keras.datasets import mnist
from progressbar import progressbar
import numpy as np
import multiprocessing as mp
from itertools import product

from layers.Conv2D import Conv2D
from layers.Pooling import Pooling
from layers.Dense import Dense

# %%
(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X, test_X = train_X / 255.0, test_X / 255.0

train_X, test_X = train_X[:10000], test_X[:1000]
train_y, test_y = train_y[:10000], test_y[:1000]

print('Train: X=%s, y=%s' % (train_X.shape, train_y.shape))
print('Test: X=%s, y=%s' % (test_X.shape, test_y.shape))

# %%
# define the model

model = [
    Conv2D(32, 2, 1, 1),
    Pooling(2, 2, 'max'),
    Dense(32*14*14, 100),
    Dense(100, 10)
]

# %%

def predict(X):
    for layer in model:
        X = layer.forward(X)
    return X

def predict_batch(X):
    with mp.Pool() as pool:
        return pool.map(predict, X)

def learn(model, grad_y_pred, lr):
    for layer in reversed(model):
        grad_y_pred = layer.backward(grad_y_pred, lr)

def learn_batch(model, grad_y_pred, lr):
    with mp.Pool() as pool:
        pool.starmap(learn, zip(model, grad_y_pred, lr))

def train(X, y, lr=0.01):
    y_pred = predict(X)
    loss = np.square(y_pred - y).sum()
    print('loss: %f' % loss)

    grad_y_pred = 2.0 * (y_pred - y)
    learn_batch(model, grad_y_pred, lr)


# %%

# train the model
train(train_X, train_y)
# %%
