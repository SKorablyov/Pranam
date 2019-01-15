import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

#
# #X = np.arange(1000.0) * 0.001
#
#
# def camel(X):
#     relu1 = np.abs((X - 0.2))
#     relu2 = np.abs((X - 0.5))
#     relu3 = np.abs((X - 0.7))
#     return relu1 - relu2 + relu3
#
#
# #plt.plot(camel(X + 0.2))
# #plt.plot(camel(X + 0.7))
#
# x = np.linspace(-1, 2, 100)
# y = np.linspace(-1, 2, 100)
# xv, yv = np.meshgrid(x, y)
#
#
# plt.plot(camel(x))
# plt.plot(camel(x - 0.2))
# plt.plot(camel(x) + camel(x-0.2))
# #plt.show()
#
# fig = plt.figure()
# #fig.set_size_inches(12.8, 12.8)
# ax = fig.gca(projection='3d')
#
# zv = np.reshape(camel(np.reshape(xv,[-1])),xv.shape) + np.reshape(camel(np.reshape(yv,[-1]) -0.2),xv.shape)
#
# surf = ax.plot_surface(xv, yv, zv ,rstride=1, cstride=1, cmap=cm.coolwarm, color='c', linewidth=0)
# plt.show()

# loss = (tf.Variable(9.0) - 2)**2
# optim = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# grads_and_vars = optim.compute_gradients(loss)
#
# grad = grads_and_vars[0][0]
# loss2 = (tf.Variable(0.0) + tf.stop_gradient(grad))**2
# optim2 = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# grads_and_vars2 = optim2.compute_gradients(loss2)
# grads_and_vars2 = [gv for gv in grads_and_vars if gv[0] is not None]


add = tf.add_n([tf.Variable([10,11,12]),tf.Variable([1000,11,12])])


sess = tf.Session()
sess.run(tf.global_variables_initializer())

print sess.run(add)