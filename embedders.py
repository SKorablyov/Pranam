import tensorflow as tf
import numpy as np


class ScaledTanh:
    def __init__(self,name,sess,init_a,train_a,init_b,train_b):
        self.sess = sess
        self.var_a = tf.get_variable(name + "_var_a", dtype=tf.float32, initializer=init_a, trainable=train_a)
        self.var_b = tf.get_variable(name + "_var_b", dtype=tf.float32, initializer=init_b, trainable=train_b)

    def doit(self,inputs):
        self.sess.run(tf.global_variables_initializer())  # FIXME initialize uninitalized vars only
        scaling = 1.7159 / tf.maximum(tf.abs(tf.reduce_max(inputs)), tf.abs(tf.reduce_min(inputs)))
        scaling_const = self.sess.run(scaling)
        act_inputs = self.var_b * tf.nn.tanh(self.var_a * scaling_const * inputs)
        return act_inputs

class FCEmbedder:
    def __init__(self,sess,variables,gradients,embed_shape,initializers,acts):
        """
        :param variables: 2D array of variables to embed
        :param gradients: 2D array of gradients for the varibles to embed
        """
        # fixme I am not sure what's the safe way to create variable without a name
        # FIXME: I need to initialize each of the weights separately
        # set up the shapes of the embedding
        assert (embed_shape[0] is None) and (embed_shape[-1] is None), "first and last layers should be None"
        embed_shape[0] = len(variables)
        varshapes = [np.asarray(v.get_shape().as_list()) for v in variables[0]]
        embed_shape[-1] = int(np.sum([np.prod(shape) for shape in varshapes]))
        print "initialized FC embedding with of shape",embed_shape

        # build embedding
        input = tf.range(0, embed_shape[0])
        embed_params = tf.get_variable("FCEmbed_0", shape=[embed_shape[0], embed_shape[1]],
                                       initializer=initializers[0])
        top_layer = tf.nn.embedding_lookup(params=embed_params, ids=input)
        for i in range(1, len(embed_shape) - 1):
            w = tf.get_variable("FCEmbed_" + str(i), shape=[embed_shape[i], embed_shape[i + 1]],
                                initializer=initializers[i])
            top_layer = tf.matmul(top_layer, w)
            if i < (len(embed_shape) - 2):
                top_layer = acts[i-1](top_layer)
        top_layer = acts[len(embed_shape) - 2](top_layer)

        # compute the gradient on embedding
        input_gradients = []
        for grads in gradients:
            input_gradients.append(tf.concat([tf.reshape(g,[-1]) for g in grads],axis=0))
        input_gradients = tf.stack(input_gradients,axis=0)
        embedding_loss = (top_layer - tf.stop_gradient(top_layer) + tf.stop_gradient(input_gradients))**2
        optim = tf.train.GradientDescentOptimizer(0.01) # todo this is a trick to get gradients; rewrite cleaner
        self.embedding_grads = []
        self.embedding_vars = []
        for grad,var in optim.compute_gradients(embedding_loss):
            if grad is not None:
                self.embedding_grads.append(grad)
                self.embedding_vars.append(var)

        # compute the state of the embedding
        slices = np.concatenate([[0],np.cumsum([int(np.prod(shape)) for shape in varshapes])],axis=0)
        slices = [np.array([slices[i], slices[i+1] - slices[i]]) for i in range(slices.shape[0]-1)]
        self.embedded_variables = [[] for _ in range(embed_shape[0])]
        for clone_num in range(embed_shape[0]):
            for var_num in range(len(varshapes)):
                sl = [slices[var_num][0], slices[var_num][1]]
                embedded_var = tf.slice(top_layer[clone_num,:],[sl[0]],[sl[1]])
                embedded_var = tf.reshape(embedded_var, varshapes[var_num])
                self.embedded_variables[clone_num].append(tf.stop_gradient(embedded_var))

    def compute_gradients(self):
        "compute and return state of embedding, and gradients on embedding"
        return self.embedded_variables, self.embedding_grads,self.embedding_vars