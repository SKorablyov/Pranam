import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import utils,embedders


# embed_shape = [1,1,8000]
# embed_d = len(embed_shape)
#
# initializers = [tf.contrib.layers.xavier_initializer(),
#                    tf.contrib.layers.xavier_initializer(),
#                    tf.contrib.layers.xavier_initializer()]
# #scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=0.01,
# #                                   train_b=False).doit  # fixme trainabe
# acts = [tf.nn.relu]
#
#
# input = tf.range(0, embed_shape[0])
# embed_params = tf.get_variable("FCEmbed_0", shape=[embed_shape[0], embed_shape[1]],
#                                initializer=initializers[0])
# tf.summary.histogram("FCEmbed_0", embed_params)
# top_layer = tf.nn.embedding_lookup(params=embed_params, ids=input)
# for i in range(1, embed_d - 1):
#     w = tf.get_variable("FCEmbed_" + str(i), shape=[embed_shape[i], embed_shape[i + 1]],
#                         initializer=initializers[i])
#     top_layer = tf.matmul(top_layer, w)
#     if i < (len(embed_shape) - 2):
#         tf.summary.histogram("FCEmbed_" + str(i), w)
#         top_layer = acts[i - 1](top_layer)
#
# sess = tf.Session()
#
# # [X,64,8000]
# # 1     0.0004864608
# # 16    0.00039805478
# # 64    0.0002483476
# # 256   9.924395e-05
#
# # [X,1,8000]
# # 1     0.00028460877
# # 64    0.00048677114
# # 256   0.0004835139
#
# _tls = []
# for i in range(1000):
#     sess.run(tf.global_variables_initializer())
#     _tl = sess.run(top_layer)
#     _tls.append(np.reshape(_tl,[-1]))
#
# _tls = np.asarray(_tls)
# print np.mean(_tls),np.var(_tls)


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
    result = 418.9829 * tf.to_float(tf.shape(x)[1]) - tf.reduce_sum(tf.sin(tf.abs(x) ** 0.5) * x,axis=1)
    return result

# def _schwefel(x, xmin=-1, xmax=1):
#     """
#     https://www.sfu.ca/~ssurjano/schwef.html
#     """
#     x = _rescale(x, xmin, xmax, -500, 500)
#     #result = x.abs().sqrt().sin().mul(x).sum(1).mul(-1).add(418.9829 * x.size(1))
#     result = 418.9829 * tf.to_float(tf.shape(x)[1]) - tf.reduce_sum(tf.sin(tf.abs(x)**0.5) * x,axis=1)
#     return result

# plot some 2d stuff:
fig = plt.figure()
fig.set_size_inches(12.8, 12.8)
ax = fig.gca(projection='3d')
_x = np.linspace(-1, 1, 100,dtype=np.float32)
_y = np.linspace(-1, 1, 100,dtype=np.float32)
_xv, _yv = np.meshgrid(_x, _y)

# zv = the_function(torch.from_numpy(np.stack([xv, yv], 1))).numpy()

zv = _schwefel(np.stack([_xv, _yv], 1))
sess = tf.Session()
_zv = sess.run(zv)
#print _zv.shape, _xv.shape


surf = ax.plot_surface(_xv, _yv, _zv, rstride=1, cstride=1, cmap=cm.coolwarm, color='c', alpha=0.3, linewidth=0)
#xys = network().detach().numpy()
#zs = the_function(network()).detach().numpy()

plt.show()
# print zs
#ax.scatter(xys[:, 0], xys[:, 1], zs, color="k", s=50)
#if not os.path.exists(os.path.join(args.save_dir, str(lr))): os.makedirs(os.path.join(args.save_dir, str(lr)))
#plt.savefig(os.path.join(args.save_dir, str(lr), "surf_" + str(i) + ".png"))
#plt.close()
