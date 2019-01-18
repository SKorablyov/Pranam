import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import utils,embedders


embed_shape = [1,1,8000]
embed_d = len(embed_shape)

initializers = [tf.contrib.layers.xavier_initializer(),
                   tf.contrib.layers.xavier_initializer(),
                   tf.contrib.layers.xavier_initializer()]
#scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=0.01,
#                                   train_b=False).doit  # fixme trainabe
acts = [tf.nn.relu]


input = tf.range(0, embed_shape[0])
embed_params = tf.get_variable("FCEmbed_0", shape=[embed_shape[0], embed_shape[1]],
                               initializer=initializers[0])
tf.summary.histogram("FCEmbed_0", embed_params)
top_layer = tf.nn.embedding_lookup(params=embed_params, ids=input)
for i in range(1, embed_d - 1):
    w = tf.get_variable("FCEmbed_" + str(i), shape=[embed_shape[i], embed_shape[i + 1]],
                        initializer=initializers[i])
    top_layer = tf.matmul(top_layer, w)
    if i < (len(embed_shape) - 2):
        tf.summary.histogram("FCEmbed_" + str(i), w)
        top_layer = acts[i - 1](top_layer)

sess = tf.Session()

# [X,64,8000]
# 1     0.0004864608
# 16    0.00039805478
# 64    0.0002483476
# 256   9.924395e-05

# [X,1,8000]
# 1     0.00028460877
# 64    0.00048677114
# 256   0.0004835139

_tls = []
for i in range(1000):
    sess.run(tf.global_variables_initializer())
    _tl = sess.run(top_layer)
    _tls.append(np.reshape(_tl,[-1]))

_tls = np.asarray(_tls)
print np.mean(_tls),np.var(_tls)
