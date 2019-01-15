import tensorflow as tf

def perceptron_embedding(sizes, sess, coord):
    # init network (first 2 layers)
    input = tf.range(0, sizes[0])

    embed_params = tf.get_variable("perc_embed", shape=[sizes[0], sizes[1]],
                         initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    # initializer=tf.contrib.layers.xavier_initializer())
    top_layer = tf.nn.embedding_lookup(params=embed_params, ids=input)
    # top_layer = tf.get_variable("embed",initializer=tf.truncated_normal(shape=[sizes[0],sizes[1]]))

    # build network
    for i in range(1, len(sizes) - 1):
        name = "perc_fc" + str(i)
        shape = [sizes[i], sizes[i + 1]]
        w = tf.get_variable(name + "_w", shape=shape,
                         initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
        # w = tf.get_variable(name+"_w", shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        # b = tf.get_variable(name+"_b", shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        top_layer = tf.matmul(top_layer, w)
        if i < (len(sizes) - 2):
            top_layer = tf.nn.relu(top_layer)

    # rescale with tg activation
    sess.run(tf.global_variables_initializer())  # FIXME: I need to initialize each of the weights !
    scaling = 1.7159 / tf.maximum(tf.abs(tf.reduce_max(top_layer)), tf.abs(tf.reduce_min(top_layer)))

    scaling_const = sess.run(scaling)
    a_ = [1.0]
    a = tf.get_variable("a", dtype=tf.float32, initializer=tf.constant(a_), trainable=True)
    out = tf.nn.tanh(tf.multiply(scaling_const * top_layer, a))
    b_ = [0.01]
    b = tf.get_variable("b", dtype=tf.float32, initializer=tf.constant(b_), trainable=True)
    out = tf.multiply(out, b)
    return out


def net3_embed(X, fun_shape, em_shape, sess, coord, tl):
    """

    :param X:
    :param fun_shape:
    :param em_shape:
    :param sess:
    :param coord:
    :param tl:
    :return:
    
def test_accuracy(X, x_test,y_test,em_shape,sess, coord):

 

    return accuracy

"""
    l0 = tf.expand_dims(X, 1)

    if tl[2] == 1:
        y = True
    else:
        y = False

    W1 = tf.get_variable("W1", shape=[fun_shape[0], fun_shape[1]],
                         initializer=tf.contrib.layers.xavier_initializer(), trainable=y)
    l1 = tf.reduce_sum(tf.expand_dims(l0, 3) * W1, axis=2)

    l1_act = tf.nn.relu(l1)

    W2 = eval("perceptron_embedding")(em_shape, sess=sess, coord=coord)
    W2 = tf.reshape(W2, [1, em_shape[0], fun_shape[1], fun_shape[2]])

    l2 = tf.reduce_sum(tf.expand_dims(l1_act, 3) * W2, axis=2)
    l2_act = tf.nn.relu(l2)
    shape = [fun_shape[2], fun_shape[3]]

    W3 = tf.get_variable("W3", shape=[fun_shape[2], fun_shape[3]],
                         initializer=tf.initializers.zeros(), trainable=y)
    l3 = tf.reduce_sum(tf.expand_dims(l2_act, 3) * W3, axis=2)
    return X, l3, [W1, W2, W3]

