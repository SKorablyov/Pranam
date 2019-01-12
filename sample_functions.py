import tensorflow as tf

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
    # result = x.abs().sqrt().sin().mul(x).sum(1).mul(-1).add(418.9829 * x.size(1))
    result = 418.9829 * tf.to_float(tf.shape(x)[1]) - tf.reduce_sum(tf.sin(tf.abs(x) ** 0.5) * x, axis=1)
    return result

