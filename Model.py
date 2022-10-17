# %%
# load the dataset

from tensorflow.keras.datasets import mnist
from progressbar import ProgressBar as progressbar
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

# reduce the size of the dataset
train_X, test_X = train_X[:10000], test_X[:1000]
train_y, test_y = train_y[:10000], test_y[:1000]

# scale the data
train_X, test_X = train_X / 255.0, test_X / 255.0

# zero-center and normalize the images
train_X = (train_X - np.mean(train_X)) / np.std(train_X)

# need the fourth dimension to represent the number of channels
train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)

print('Train: X=%s, y=%s' % (train_X.shape, train_y.shape))
print('Test: X=%s, y=%s' % (test_X.shape, test_y.shape))

# %%
# define the model
model = []

model.append(Conv2D(32, 2, 1, 1, activation='relu'))
model.append(Pooling(2, 2, 'max'))
model.append(Flatten())

# determine the number of input features by running one forward pass on one image
dims = train_X[0]
dims = dims.reshape(1, *dims.shape)
for layer in model:
    dims = layer.forward(dims)

model.append(Dense(np.prod(dims.shape[1:]), 128, activation='relu'))
model.append(Dense(128, 10, activation='relu'))

for layer in model[3:]:
    dims = layer.forward(dims)

# %%


def predict(X, model):
    # forward pass on a single image
    for layer in model:
        X = layer.forward(X)
    return X


def train(X, y, model, lr=1e-4, epochs=10):

    # need to make epochs work
    # need to do forward passes chunks of mp.cpu_count() images at a time
    # when each forward pass is done, do a backward pass on the same chunk of images
    loss = 0
    chunksize = 100
    assert (X.shape[0] % chunksize == 0)

    for epoch in range(epochs):
        acclist = []
        losslist = []
        p = progressbar(
            max_value=X.shape[0], prefix=f'epoch {epoch}/{epochs} ', redirect_stdout=True)
        # break into chunks of at most mp.cpu_count() images
        for i in range(0, len(X), chunksize):
            # forward pass
            y_pred = predict(X[i:min(X.shape[0], i+chunksize)], model)

            # gradient
            grad_y_pred = y_pred - \
                np.eye(10)[y[i:min(X.shape[0], i+chunksize)]]
            acc = np.mean(np.argmax(y_pred, axis=1) == y[i:i+chunksize])
            acclist.append(acc)

            loss = np.square(grad_y_pred).sum()
            losslist.append(loss)
            if loss is np.nan:
                raise Exception('Loss is NaN')
                exit(-1)
            # print(f'loss={loss.round(2)}, acc={acc * 100.0}%')

            # backward pass
            for layer in reversed(model):
                grad_y_pred = layer.backward(grad_y_pred, lr)

            p.update(i)
        p.finish()
        print(f'loss = {np.mean(losslist)}   accuracy = {np.mean(acclist)}%')


# %%
# train the model
train(train_X, train_y, model)


# %%

# test the accuracy of the model

def test(X, y, model):
    y_pred = np.zeros(y.shape)
    for i in range(0, len(X), 100):
        y_pred[i:min(X.shape[0], i+100)] = np.argmax(
            predict(X[i:min(X.shape[0], i+100)], model), axis=1)
    return np.mean(y_pred == y)

print('Test accuracy: %.2f%%' % (test(test_X, test_y, model) * 100))