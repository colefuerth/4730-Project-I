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
from layers.Flatten import Flatten

# %%
# import the data
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# scale the data
train_X, test_X = train_X / 255.0, test_X / 255.0

# reduce the size of the dataset
train_X, test_X = train_X[:10000], test_X[:1000]
train_y, test_y = train_y[:10000], test_y[:1000]

# need the fourth dimension to represent the number of channels
train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)

print('Train: X=%s, y=%s' % (train_X.shape, train_y.shape))
print('Test: X=%s, y=%s' % (test_X.shape, test_y.shape))

# %%
# define the model

model = []
model.append(Conv2D(32, 2, 1, 1))
model.append(Pooling(2, 2, 'max'))
model.append(Flatten())
dims = train_X[0]
for layer in model:
    # print(dims.shape)
    dims = layer.forward(dims)
# print(dims.shape)
model.append(Dense(dims.shape[0], 128))
model.append(Dense(128, 10))


# %%
# define functions for training

# takes a tuple (i, X)
# multithreading happens asynchronously so we need to retain the order of predictions when they come back out the other side
def predict(X) -> np.ndarray:
    # forward pass on a single image
    for layer in model:
        X = layer.forward(X)
    return X

def predict_batch(X) -> np.ndarray:
    # forward pass on a batch of images
    with mp.Pool(mp.cpu_count()) as p:
        ps = [p.apply_async(predict, args=(x,)) for x in X]
        y_pred = [p.get() for p in progressbar(ps, prefix='predicting ')]

def train(X, y, lr=0.01):
    # forward pass
    y_pred = predict_batch(X)
    # calculate loss
    loss = np.square(y_pred - y).sum()
    print('loss: %f' % loss)

    # backward pass
    grad_y_pred = 2.0 * (y_pred - y)
    for layer in reversed(model):
        grad_y_pred = layer.backward(grad_y_pred, lr)


# %%

# train the model
train(train_X, train_y)
# %%
