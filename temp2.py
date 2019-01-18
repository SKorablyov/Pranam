import tensorflow as tf
import inputs
import numpy as np

b_size =1
initializers = [tf.contrib.layers.xavier_initializer(),
                tf.contrib.layers.xavier_initializer(),
                tf.contrib.layers.xavier_initializer()]
trainables = [True,True,True]

# load dataset and only create one loader for all copies
if not "mnist_fcnet_loader" in globals().keys():
    globals()["mnist_fcnet_loader"] = inputs.load_mnist(b_size)
b_trX, b_trY, b_teX, b_teY = globals()["mnist_fcnet_loader"]
# initialize variables
with tf.variable_scope("mnist_fcnet"):
    w1 = tf.get_variable("w1", shape=[784, 128], initializer=initializers[0], trainable=trainables[0])
    w2 = tf.get_variable("w2", shape=[128, 64], initializer=initializers[1], trainable=trainables[1])
    w3 = tf.get_variable("w3", shape=[64, 10], initializer=initializers[2], trainable=trainables[2])
# build network
tr_input = tf.reshape(b_trX, [b_size, -1])
tr_logits = tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(tr_input, w1)), w2)), w3)

sess = tf.Session()
coord = tf.train.Coordinator()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess,coord)

# 1280 0.06991143
# 128 0.12277809, 0.105972305, 0.11091379, 0.10604664
# 12  0.02434467, 0.029673038, 0.029101579, 0.030923652

# w2: 0.0104190465

# _tlogits = []
# for i in range(1000):
#     sess.run(tf.global_variables_initializer())
#
#     _tlogit = sess.run(w2)
#     _tlogits.append(np.reshape(_tlogit,[-1]))
#
# _tlogits = np.asarray(_tlogits)
# print _tlogits
# print np.mean(_tlogits), np.var(np.reshape(_tlogits,[-1]))


print np.var(2*np.array([2,4,9]))