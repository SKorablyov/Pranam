import time
import tensorflow as tf
import numpy as np
from pranam import PranamOptimizer
import embedders
import networks
import utils

from config import cfg1

#opt = PranamOptimizer(sess=sess,func=network,func_pars=[0],embed_vars=["a:0","b:0"],batch_size=5,learning_rate=0.01)


cfg = cfg1()

opt = PranamOptimizer(sess=cfg.sess,func=cfg.func,func_pars=cfg.func_pars,num_clones=cfg.num_clones,
                      optimizer=cfg.optimizer,learning_rate=cfg.learning_rate,batch_size=cfg.batch_size,
                      embed_vars=cfg.embed_vars,embedder=cfg.embedder,embedder_pars=cfg.embedder_pars)


train_op,metrics = opt.train_step()
cost_mean = tf.reduce_mean(metrics[0], axis=0)
cost_min = tf.reduce_min(metrics[0], axis=0)
best_guess = metrics[1][tf.argmin(metrics[0])]

# init_vars = sess.run(tf.report_uninitialized_variables())
# init_vars = [tf.get_variable(var) for var in init_vars]
# sess.run(tf.initialize_variables(init_vars))

cfg.sess.run(tf.global_variables_initializer()) # fixme only initialize uninitialized variables

np.set_printoptions(precision=2)
for i in range(5000):
    _, _cost_mean,_cost_min,_best_guess = cfg.sess.run([train_op, cost_mean, cost_min, best_guess])
    print "step:", i, "cost_mean:", _cost_mean, "cost_min:",_cost_min

    if i %100 ==1:
        print "guess:",_best_guess
