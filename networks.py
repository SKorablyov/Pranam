import tensorflow as tf
import numpy as np
import time
import inputs
from math import pi
from math import exp


def dummy_net(just_anything):
    """ Dummy net to test if GD works.
    :param just_anything: a placeholder which does not matter
    :return:
    """
    dummy1 = tf.Variable(10)
    a = tf.get_variable("a",shape=[1,3,5,5], initializer=tf.initializers.random_uniform())
    b = tf.get_variable("b",shape=[], initializer=tf.initializers.random_uniform())
    c = tf.get_variable("c",shape=[],initializer=tf.initializers.random_uniform())
    d = tf.get_variable("d",shape=[],initializer=tf.initializers.random_uniform())
    z = (a + b) * (c + d)
    cost = (z - 7)**2
    dummy2 = tf.Variable(20)
    return cost, [tf.reduce_sum(cost)]


def rosen_net(dim=10, function="_rosenbrock"):

    def _rescale(x, a, b, c, d):
        """
        Rescales variable from [a, b] to [c, d]
        """
        return c + ((d - c) / (b - a)) * (x - a)

    def _ROSENBROCK(x, xmin=-1, xmax=1):
        #DON`T WORK!
        """
        https://www.sfu.ca/~ssurjano/rosen.html
        """
        data_tensor = _rescale(x, xmin, xmax, -2.048, 2.048)
        columns = tf.unstack(data_tensor)

        # Track both the loop index and summation in a tuple in the form (index, summation)
        index = tf.constant(1)
        summation = tf.zeros(tf.shape(x))

        # The loop condition, note the loop condition is 'i < n-1'
        def condition(index, summation):
            return tf.less(index, tf.subtract(tf.shape(columns)[0], 1))

            # The loop body, this will return a result tuple in the same form (index, summation)

        def body(index, summation):
            x_i = tf.gather(columns, index)
            x_ip1 = tf.gather(columns, tf.add(index, 1))

            first_term = (x_ip1 - x_i ** 2) ** 2
            second_term = (x_i - 1) ** 2
            summand = tf.add(tf.multiply(100.0, first_term), second_term)

            return tf.add(index, 1), tf.add(summation, summand)

            # We do not care about the index value here, return only the summation

        result = tf.reduce_mean(tf.while_loop(condition, body, [index, summation])[1], axis=1)
        return result

    guess = tf.get_variable(name="rosennet_guess", shape=[dim], initializer=tf.random_uniform_initializer())
    cost = eval(function)(guess)
    return cost, [cost, guess]


def michal_net(dim=10, function="_mihalevich"):

    def _rescale(x, a, b, c, d):
        """
        Rescales variable from [a, b] to [c, d]
        """
        return c + ((d - c) / (b - a)) * (x - a)

    def _MICHALEWICZ(x, xmin=-1, xmax=1):
        # DON`T WORK !
        """
        https://www.sfu.ca/~ssurjano/michal.html
        """
        data_tensor = _rescale(x, xmin, xmax, 0, pi)
        columns = tf.unstack(data_tensor)

        # Track both the loop index and summation in a tuple in the form (index, summation)

        index = tf.constant(1)
        summation = tf.zeros(tf.shape(x))

        # The loop condition, note the loop condition is 'i < n-1'
        def condition(index, summation):
            return tf.less(index, tf.subtract(tf.shape(columns)[0], 1))

            # The loop body, this will return a result tuple in the same form (index, summation)

        def body(index, summation):
            x_i = tf.gather(columns, index)
            mult_1 = tf.sin(x_i)
            i = tf.to_float(index)
            mult2 = tf.sin((i * x_i ** 2) / pi)
            mult_2 = mult2 ** 20

            summand = tf.add(-1 * mult_1, mult_2)

            return tf.add(index, 1), tf.add(summation, summand)

            # We do not care about the index value here, return only the summation

        result = tf.while_loop(cond=condition, body=body, loop_vars=[index, summation])[1]
        result = tf.reduce_mean(result, axis=1)

        return result

    guess = tf.get_variable(name="rosennet_guess", shape=[dim], initializer=tf.random_uniform_initializer())
    cost = eval(function)(guess)
    return cost, [cost, guess]


def rastrigin_net(dim=10, function="_rastrigin"):

    def _rescale(x, a, b, c, d):
        """
        Rescales variable from [a, b] to [c, d]
        """
        return c + ((d - c) / (b - a)) * (x - a)

    def _rastrigin(x, xmin=-1, xmax=1):
        """
        https://www.sfu.ca/~ssurjano/rastr.html
        """
        x = _rescale(x, xmin, xmax, -5.12, 5.12)

        result = 10.0 * tf.to_float(tf.shape(x)[0]) + tf.reduce_sum(x ** 2 - 10 * (tf.cos(2 * pi * x)), axis=1)
        return result

    guess = tf.get_variable(name="rastrnet_guess", shape=[dim], initializer=tf.random_uniform_initializer())
    cost = eval(function)(guess)
    return cost, [cost, guess]


def stybtang_net(dim=10, function="_stiblinsky_tang"):

    def _rescale(x, a, b, c, d):
        """
        Rescales variable from [a, b] to [c, d]
        """
        return c + ((d - c) / (b - a)) * (x - a)

    def _stiblinsky_tang(x, xmin=-1, xmax=1):
        """
        https://www.sfu.ca/~ssurjano/stybtang.html
        """
        x = _rescale(x, xmin, xmax, -5, 5)

        result = 0.5 * tf.reduce_sum(x ** 4 - 16 * x ** 2 + 5 * x, axis=1)
        return result

    guess = tf.get_variable(name="stybtangnet_guess", shape=[dim], initializer=tf.random_uniform_initializer())
    cost = eval(function)(guess)
    return cost, [cost, guess]

def sphere_net(dim=10, function="_sphere"):

    def _rescale(x, a, b, c, d):
        """
        Rescales variable from [a, b] to [c, d]
        """
        return c + ((d - c) / (b - a)) * (x - a)

    def _sphere(x, xmin=-1, xmax=1):
        """
        https://www.sfu.ca/~ssurjano/spheref.html
        """
        x = _rescale(x, xmin, xmax, -5, 5)

        result = tf.reduce_sum(x ** 2, axis=1)
        return result

    guess = tf.get_variable(name="spherenet_guess", shape=[dim], initializer=tf.random_uniform_initializer())
    cost = eval(function)(guess)
    return cost, [cost, guess]

def ackley_net(dim=10, function="_ackley"):

    def _rescale(x, a, b, c, d):
        """
        Rescales variable from [a, b] to [c, d]
        """
        return c + ((d - c) / (b - a)) * (x - a)

    def _ackley(x, xmin=-1, xmax=1):
        """
        https://www.sfu.ca/~ssurjano/ackley.html
        """
        a = 20
        b = 0.2
        c = 2 * pi

        d = tf.to_float(tf.shape(x)[0])
        d = 1 / d
        print (d)

        x = _rescale(x, xmin, xmax, -32.768, 32.768)

        term1 = -1 * a * tf.math.exp(-b * (d * tf.reduce_sum(x ** 2, axis=1)) ** 0.5)
        term2 = -1 * tf.math.exp(d * tf.reduce_sum(tf.cos(c * x), axis=1))

        return term1 + term2 + a + exp(1)

    guess = tf.get_variable(name="ackleynet_guess", shape=[dim], initializer=tf.random_uniform_initializer())
    cost = eval(function)(guess)
    return cost, [cost, guess]






def schwefel_net(dim=10, function="_schwefel"):
    """ Example to test nonconvex optimization. The network itself is one layer of size dim. The loss function is
    also of dimention dim, and might be typically non-convex.

    :param dim: dimentionality of the Schwefel function
    :return:
    """
    def _rescale(x, a, b, c, d):
        """
        Rescales variable from [a, b] to [c, d]
        """
        return c + ((d - c) / (b - a)) * (x - a)

    def _schwefel(x, xmin=-1, xmax=1):
        """
        https://www.sfu.ca/~ssurjano/schwef.html
        """
        x = _rescale(x, xmin, xmax, -500, 500)
        result = 418.9829 * tf.to_float(tf.shape(x)[0]) - tf.reduce_sum(tf.sin(tf.abs(x) ** 0.5) * x)
        return result

    guess = tf.get_variable(name="schnet_guess", shape=[dim], initializer=tf.random_uniform_initializer())
    cost = eval(function)(guess)
    return cost, [cost, guess]



def mnist_fcnet(b_size,initializers,trainables,shapes,acts):
    """ Network from tensorflow's first tutorial on MNIST
    :param b_size: integer, internal batch size
    :return:
    """
    # load dataset and only create one loader for all copies
    if not "mnist_fcnet_loader" in globals().keys():
        globals()["mnist_fcnet_loader"] = inputs.load_mnist(b_size)
    b_trX, b_trY, b_teX, b_teY = globals()["mnist_fcnet_loader"]
    # initialize variables
    with tf.variable_scope("mnist_fcnet"):
        w1 = tf.get_variable("w1", shape=[784, shapes[0]], initializer=initializers[0], trainable=trainables[0])
        w2 = tf.get_variable("w2", shape=[shapes[0], shapes[1]], initializer=initializers[1], trainable=trainables[1])
        w3 = tf.get_variable("w3", shape=[shapes[1], 10], initializer=initializers[2], trainable=trainables[2])
    # build network
    tr_input = tf.reshape(b_trX, [b_size, -1])
    tr_logits = tf.matmul(acts[1](tf.matmul(acts[0](tf.matmul(tr_input, w1)), w2)), w3)
    tr_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tr_logits, labels=b_trY),axis=0)
    tr_preds = tf.one_hot(tf.argmax(tf.nn.softmax(tr_logits), axis=1), 10,dtype=tf.float32)
    tr_acc = tf.reduce_mean(tf.reduce_sum(tf.one_hot(b_trY,10,dtype=tf.float32) * tr_preds,axis=1),axis=0)
    te_input = tf.reshape(b_teX, [b_size, -1])
    te_logits = tf.matmul(acts[1](tf.matmul(acts[0](tf.matmul(te_input, w1)), w2)), w3)
    te_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=te_logits, labels=b_teY),axis=0)
    te_preds = tf.one_hot(tf.argmax(tf.nn.softmax(te_logits), axis=1), 10,dtype=tf.float32)
    te_acc = tf.reduce_mean(tf.reduce_sum(tf.one_hot(b_teY,10,dtype=tf.float32) * te_preds,axis=1),axis=0)
    # summaries
    tf.summary.histogram("w1",w1)
    tf.summary.histogram("w2",w2)
    tf.summary.histogram("w3",w3)
    return tr_loss,[tr_loss,tr_acc,te_loss,te_acc]


