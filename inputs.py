import tensorflow as tf
import numpy as np
import time, os, sys, socket

from keras.datasets import mnist  # subroutines for fetching the MNIST dataset
from keras.utils import np_utils  # utilities for one-hot encoding of ground truth values



def load_mnist(b_size):
    # load mnist
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = np.asarray(x_train / 255.0), x_test / 255.0
    # build input queues
    tr_q = tf.train.slice_input_producer([tf.convert_to_tensor(x_train, dtype=tf.float32),
                                          tf.convert_to_tensor(y_train, tf.int32)])
    b_trX, b_trY = tf.train.shuffle_batch(tr_q, num_threads=1, batch_size=b_size, capacity=64 * b_size,
                                          min_after_dequeue=32 * b_size, allow_smaller_final_batch=False)
    te_q = tf.train.slice_input_producer([tf.convert_to_tensor(x_test, tf.float32),
                                          tf.convert_to_tensor(y_test, tf.int32)])
    b_teX, b_teY = tf.train.shuffle_batch(te_q, num_threads=1, batch_size=b_size, capacity=64 * b_size,
                                          min_after_dequeue=32 * b_size, allow_smaller_final_batch=False)
    return b_trX, b_trY, b_teX, b_teY


def hierarchical_compositional(genf_shape, n_samples, W=None, noise=None):
    "generate a bunch of samples, the activation function is frozen as relu"
    X = np.matrix(np.asarray(np.random.uniform(size=[n_samples, genf_shape[0]])))
    lastX = X
    # initialize
    if W is None:
        W = []
        for i in np.arange(len(genf_shape) - 1) + 1:
            w = np.matrix(np.random.uniform(low=-2, high=2, size=[genf_shape[i - 1], genf_shape[i]]))
            W.append(w)
    # apply weights, get Ys
    for i in np.arange(len(genf_shape) - 1):
        print("layer", i, "incoming shape", lastX.shape)
        # print "test shape", lastX.shape,W[i].shape, (lastX * W[i]).shape, np.maximum(lastX * W[i],0).shape
        lastX = np.maximum(np.asarray(lastX * W[i]), 0)
    Y = lastX
    if noise is not None:
        Y = Y + np.random.uniform(size=Y.shape, low=-0.5 * noise, high=0.5 * noise)
    return X, Y, W


def generate_dataset(out_path, f_shape, train_samples, test_samples, noise):
    if not os.path.exists(out_path): os.makedirs(out_path)
    op = str(out_path)
    op = op[-5:]

    if op == "mnist":
        num_train = 60000  # there are 60000 training examples in MNIST
        num_test = 10000  # there are 10000 test examples in MNIST

        height, width, depth = 28, 28, 1  # MNIST images are 28x28 and greyscale
        num_classes = 10  # there are 10 classes (1 per digit)

        (X_train, y_train), (X_test, y_test) = mnist.load_data()  # fetch MNIST data

        X_train = X_train.reshape(num_train, height * width)  # Flatten data to 1D
        X_test = X_test.reshape(num_test, height * width)  # Flatten data to 1D
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        #X_train /= 255  # Normalise data to [0, 1] range
        #X_test /= 255  # Normalise data to [0, 1] range
        Y_train = np_utils.to_categorical(y_train, num_classes)  # One-hot encode the labels
        Y_test = np_utils.to_categorical(y_test, num_classes)  # One-hot encode the labels
        print("data has been saved")
        np.save(os.path.join(out_path, "X_test"), np.asarray(X_test, np.float32))
        np.save(os.path.join(out_path, "Y_test"), np.asarray(Y_test, np.float32))
        np.save(os.path.join(out_path, "X_train"), np.asarray(X_train, np.float32))
        np.save(os.path.join(out_path, "Y_train"), np.asarray(Y_train, np.float32))
    else:
        # generate training set
        X_train, Y_train, W = hierarchical_compositional(f_shape, n_samples=train_samples, noise=noise)
        np.save(os.path.join(out_path, "X_train"), np.asarray(X_train, np.float32))
        np.save(os.path.join(out_path, "Y_train"), np.asarray(Y_train, np.float32))
        # generate testing set
        X_test, Y_test, _ = hierarchical_compositional(f_shape, n_samples=test_samples, W=W, noise=noise)
        np.save(os.path.join(out_path, "X_test"), np.asarray(X_test, np.float32))
        np.save(os.path.join(out_path, "Y_test"), np.asarray(Y_test, np.float32))
        print("data has been saved")


def read_dataset(db_path, batch_size):
    # load dataset
    machine = socket.gethostname()
    if machine == "viacheslav-HP-Pavilion-Notebook":

        path = str(db_path)
        path = path[2:]
        path = path[:-1]
        X_train = np.load(os.path.join(path, "X_train.npy"))
        Y_train = np.load(os.path.join(path, "Y_train.npy"))
        X_test = np.load(os.path.join(path, "X_test.npy"))
        Y_test = np.load(os.path.join(path, "Y_test.npy"))
    else:
        X_train = np.load(os.path.join(db_path, "X_train.npy"))
        Y_train = np.load(os.path.join(db_path, "Y_train.npy"))
        X_test = np.load(os.path.join(db_path, "X_test.npy"))
        Y_test = np.load(os.path.join(db_path, "Y_test.npy"))

    print("loaded dataset of shapes:", X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    # make batches
    x_train, y_train = tf.train.slice_input_producer([X_train, Y_train])
    x_train, y_train = tf.train.batch([x_train, y_train], batch_size)
    x_test, y_test = tf.train.slice_input_producer([X_test, Y_test])
    x_test, y_test = tf.train.batch([x_test, y_test], batch_size)
    return x_train, y_train, x_test, y_test

