import tensorflow as tf
import inputs
import numpy as np
import time
#in embedding.py
# [num_points,depth, point_dim ] 2,16,1
shape = [2,16,100]
W1 = tf.get_variable("W1", shape=[shape[0],shape[1]], initializer=tf.ones_initializer(), dtype=tf.float64) / shape[1]
W2 = tf.get_variable("W2", shape=[shape[1],shape[2]], initializer=tf.ones_initializer(), dtype=tf.float64) / shape[2]

act = tf.matmul(W1,W2)
cost = tf.reduce_mean(tf.reduce_mean((act - 0.5)**2,axis=1),axis=0)

opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
#train_step = opt.minimize(cost)
grads, vars = zip(*opt.compute_gradients(cost))
grad_w1 = tf.zeros_like(grads[0]) #* shape[1]
grad_w2 = grads[1] #* shape[0] #* shape[1] #* shape[2]
train_step = opt.apply_gradients(zip([grad_w1,grad_w2],vars))


sess = tf.Session()
sess.run(tf.global_variables_initializer())


for i in range(200):
    print sess.run([(act-0.5)])
