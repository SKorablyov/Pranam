import tensorflow as tf


def init_unitialized_vars(sess):
    pass


class ScaledTanh:
    def __init__(self,name,sess,init_a,train_a,init_b,train_b):
        self.sess = sess
        self.var_a = tf.get_variable(name + "_var_a", dtype=tf.float32, initializer=init_a,trainable=train_a)
        self.var_b = tf.get_variable(name + "_var_b", dtype=tf.float32, initializer=init_b,trainable=train_b)

    def doit(self,inputs):
        self.sess.run(tf.global_variables_initializer())  # FIXME initialize uninitalized vars only
        scaling = 1.7159 / tf.maximum(tf.abs(tf.reduce_max(inputs)), tf.abs(tf.reduce_min(inputs)))
        scaling_const = self.sess.run(scaling)
        act_inputs = self.var_b * tf.nn.tanh(self.var_a * scaling_const * inputs)
        return act_inputs
