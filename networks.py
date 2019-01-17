import tensorflow as tf
import time

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


def nonconvex_net(dim=10,function="_schwefel"):
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

    guess = tf.get_variable(name="schnet_guess",shape=[dim],initializer=tf.random_uniform_initializer())
    cost = eval(function)(guess)
    return cost,[cost,guess]


def mnist_fcnet(b_size,initializers,trainables):
    """ Network from tensorflow's first tutorial on MNIST
    :param b_size: integer, internal batch size
    :return:
    """
    # load mnist
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # build input queues
    tr_q = tf.train.slice_input_producer([tf.convert_to_tensor(x_train, tf.float32),
                                          tf.convert_to_tensor(y_train, tf.int32)])
    b_trX, b_trY = tf.train.shuffle_batch(tr_q, num_threads=1, batch_size=b_size, capacity=64 * b_size,
                                          min_after_dequeue=32 * b_size, allow_smaller_final_batch=False)
    te_q = tf.train.slice_input_producer([tf.convert_to_tensor(x_test, tf.float32),
                                          tf.convert_to_tensor(y_test, tf.int32)])
    b_teX, b_teY = tf.train.shuffle_batch(te_q, num_threads=1, batch_size=b_size, capacity=64 * b_size,
                                          min_after_dequeue=32 * b_size, allow_smaller_final_batch=False)

    # initialize variables
    with tf.variable_scope("mnist_fcnet"):
        w1 = tf.get_variable("w1", shape=[784, 128], initializer=initializers[0],trainable=trainables[0])
        w2 = tf.get_variable("w2", shape=[128, 64], initializer=initializers[1],trainable=trainables[1])
        w3 = tf.get_variable("w3", shape=[64, 10], initializer=initializers[2],trainable=trainables[2])

    # build network
    tr_input = tf.reshape(b_trX, [b_size, -1])
    tr_logits = tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(tr_input, w1)), w2)), w3)
    tr_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tr_logits, labels=b_trY)
    tr_preds = tf.one_hot(tf.argmax(tf.nn.softmax(tr_logits), axis=1), 10,dtype=tf.float32)
    tr_acc = tf.reduce_mean(tf.reduce_sum(tf.one_hot(b_trY,10,dtype=tf.float32) * tr_preds,axis=1),axis=0)
    te_input = tf.reshape(b_teX, [b_size, -1])
    te_logits = tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(te_input, w1)), w2)), w3)
    te_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=te_logits, labels=b_teY)
    te_preds = tf.one_hot(tf.argmax(tf.nn.softmax(te_logits), axis=1), 10,dtype=tf.float32)
    te_acc = tf.reduce_mean(tf.reduce_sum(tf.one_hot(b_teY,10,dtype=tf.float32) * te_preds,axis=1),axis=0)
    return tr_loss,[tr_loss,tr_acc,te_loss,te_acc]


