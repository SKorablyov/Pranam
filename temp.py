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


# add = tf.add_n([tf.Variable([10,11,12]),tf.Variable([1000,11,12])])
#
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# print sess.run(add)


def mnist_fcnet(b_size=1):
    """
    Network from tensorflow's first tutorial on MNIST
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
        w1 = tf.get_variable("w1", shape=[784, 128], initializer=tf.truncated_normal_initializer)
        w2 = tf.get_variable("w2", shape=[128, 64], initializer=tf.truncated_normal_initializer)
        w3 = tf.get_variable("w3", shape=[64, 10], initializer=tf.truncated_normal_initializer)

    # build network
    tr_input = tf.reshape(b_trX, [1, -1])
    tr_logits = tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(tr_input, w1)), w2)), w3)
    tr_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tr_logits, labels=b_trY)
    tr_preds = tf.argmax(tf.nn.softmax(tr_logits), axis=1)
    tr_acc = tf.reduce_mean(tf.to_float(tr_preds == b_trY))

    te_input = tf.reshape(b_teX, [1, -1])
    te_logits = tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(te_input, w1)), w2)), w3)
    te_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=te_logits, labels=b_teY)
    te_preds = tf.argmax(tf.nn.softmax(te_logits), axis=1)
    te_acc = tf.reduce_mean(tf.to_float(te_preds == b_teY))
    return tr_loss,[tr_loss,tr_acc,te_loss,te_acc]



stuff = mnist_fcnet()

sess = tf.Session()
coord = tf.train.Coordinator()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess,coord)

print sess.run([stuff])


