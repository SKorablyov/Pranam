import tensorflow as tf

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
