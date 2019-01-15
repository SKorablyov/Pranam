import tensorflow as tf
from pranam import PranamOptimizer
from networks import dummy_net,nonconvex_net


sess = tf.Session()
# test regular gradient convergence


#opt = PranamOptimizer(sess=sess,func=network,func_pars=[0],embed_vars=["a:0","b:0"],batch_size=5,learning_rate=0.01)

opt = PranamOptimizer(sess=sess,func=nonconvex_net, num_clones=25, optimizer=tf.train.GradientDescentOptimizer,
                      func_pars=[2], embed_vars=None, batch_size=1,
                      embedder_pars=[[None,25,None],
[tf.random_normal_initializer,tf.random_uniform_initializer,tf.random_uniform_initializer]], learning_rate=1e-5)


train_op,metrics = opt.train_step()
cost_mean = tf.reduce_mean(metrics[0], axis=0)
cost_min = tf.reduce_min(metrics[0], axis=0)

#init_vars = sess.run(tf.report_uninitialized_variables())
# init_vars = [tf.get_variable(var) for var in init_vars]
#sess.run(tf.initialize_variables(init_vars))

sess.run(tf.global_variables_initializer()) # fixme only initialize uninitialized variables


for i in range(5000):
    print "step:", i,sess.run([train_op,cost_mean,cost_min])