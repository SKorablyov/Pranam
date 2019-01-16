import tensorflow as tf
import networks,embedders,utils

class cfg1:
    def __init__(self):
        "network for schwefel -- needs fixes since runs out of range"
        self.name = "cfg1"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.nonconvex_net
        self.func_pars = [10]
        self.num_clones = 50
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 4e-6
        self.batch_size = 1
        # embedding pars
        self.embed_vars = None
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 50, None]
        embedding_inits = [tf.random_normal_initializer,tf.random_uniform_initializer,tf.random_uniform_initializer]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit
        embedding_acts = [tf.nn.relu,scaled_tanh]
        self.embedder_pars = [embedding_shape,embedding_inits,embedding_acts]



class cfg2:
    def __init__(self):
        self.name = "cfg2"
        self.out_path = "./results"
        self.b_size = 100
        self.num_epochs = 50


class cfg3:
    def __init__(self):
        self.name = "cfg3"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        self.func_pars = [1]
        self.num_clones = 2#5 # fixme 256
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 50
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 256, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=0.01, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.relu, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts]

