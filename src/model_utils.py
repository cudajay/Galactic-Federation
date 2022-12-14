import numpy as np
import os
import random
from tensorflow import keras
import argparse
from time import time

parser = argparse.ArgumentParser(description='Add something to describe args needs')
parser.add_argument('-p', '--predict', action=argparse.BooleanOptionalAction)
parser.add_argument('-t', '--train', action=argparse.BooleanOptionalAction)


def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func


@timer_func
def train():
    model = keras.models.load(f"working.h5")
    X = np.load("X_train.npy")
    y = np.load("y_train.npy")
    last_hist = model.fit(X, y,
                          batch_size=70,
                          epochs=1,
                          validation_split=.2,
                          verbose=True)
    print(last_hist)
    model.save(f"working.h5")


@timer_func
def predict():
    X = np.load("X_test.npy")
    y = np.load("y_test.npy")
    model = keras.models.load(f"working.h5")
    preds = model.predict(X)
    err = np.mean(keras.metrics.mae(y, preds))
    print("test error: ", err)


if __name__ == '__main__':
    if parser.predict:
        train()
    elif parser.train:
        predict()
    else:
        print("Not Option seleceted")
