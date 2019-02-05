import tensorflow as tf
import inputs
import numpy as np
import math
import time
from  networks import schwefel_net

def rescale(x, a, b, c, d):
    """
    Rescales variable from [a, b] to [c, d]
    """
    return c + ((d - c) / (b - a)) * (x - a)


def schwefel(x, xmin=-1, xmax=1):
    """
    https://www.sfu.ca/~ssurjano/schwef.html
    """
    x = rescale(x, xmin, xmax, -500, 500)
    result = 418.9829 * tf.to_float(tf.shape(x)[0]) - tf.reduce_sum(tf.sin(tf.abs(x) ** 0.5) * x)
    return result

def ackley(x, xmin=-1, xmax=1):
    """
    https://www.sfu.ca/~ssurjano/ackley.html
    """
    a = 20.0
    b = 0.2
    c = 2 * np.pi
    x = rescale(x, xmin, xmax, -32.768, 32.768)
    term1 = -a * tf.exp(-b * tf.reduce_mean(x**2)**0.5)
    term2 = -tf.exp(tf.reduce_mean(tf.cos(c*x)))
    return term1 + term2 + a + tf.exp(1.0)
#in embedding.py
# [num_points,depth, point_dim ] 2,16,1


def fsin(x):
    return 1-tf.math.cos(x-1)/(tf.abs(x-1)+1)

def calc(shape,lr,sess):
    # guess = tf.get_variable(name="schnet_guess", shape=[shape[0]], initializer=tf.random_uniform_initializer())

    # W0 = tf.get_variable("W0",shape=[1,shape[0]], initializer=tf.initializers.variance_scaling(scale=10.0, distribution="uniform"), dtype=tf.float32)
    W1 = tf.get_variable("W1", shape=[shape[0], shape[1]],
                         initializer=tf.initializers.variance_scaling(scale=10.0, distribution="uniform"),
                         dtype=tf.float32) / shape[1]
    W2 = tf.get_variable("W2", shape=[shape[1], shape[2]],
                         initializer=tf.initializers.variance_scaling(scale=10.0, distribution="uniform"),
                         dtype=tf.float32) / shape[2]

    #act = tf.matmul(W1, W2)
    # act = tf.nn.tanh(tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(W0),W1)),W2))
    act = tf.nn.tanh(tf.matmul(tf.nn.relu(W1),W2))

    # cost = tf.reduce_mean(tf.math.sin(act))
    cost =tf.reduce_mean(fsin(act))

    # cost = tf.reduce_mean(tf.reduce_mean((act + 0.3)**2,axis=1),axis=0)

    opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
    #train_step = opt.minimize(cost)
    grads, vars = zip(*opt.compute_gradients(cost))
    grad_w1 = grads[0] * shape[1] * shape[0]
    grad_w2 = grads[1] * shape[1]
    train_step = opt.apply_gradients(zip([grad_w1, grad_w2], vars))


    sess.run(tf.global_variables_initializer())
    result = []
    coords = []

    for i in range(2000):
        _, printed_value, _act = sess.run([train_step, cost, act])
        result.append(printed_value)
        coords.append(act)
        print printed_value#,_act

    return result



