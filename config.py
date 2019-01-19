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
        self.sess = tf.Session()
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
        self.num_clones = 2 # fixme 256
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

class cfg4:
    def __init__(self):
        self.name = "cfg4"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        self.func_pars = [1]
        self.num_clones = 4 # fixme 256
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



class cfg5:
    def __init__(self):
        self.name = "cfg5"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        self.func_pars = [1]
        self.num_clones = 8
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


class cfg6:
    def __init__(self):
        self.name = "cfg6"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        self.func_pars = [1]
        self.num_clones = 16
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



class cfg7:
    def __init__(self):
        self.name = "cfg7"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        self.func_pars = [1]
        self.num_clones = 32
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


class cfg8:
    def __init__(self):
        self.name = "cfg8"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        self.func_pars = [1]
        self.num_clones = 64
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

class cfg9:
    def __init__(self):
        self.name = "cfg9"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        self.func_pars = [1]
        self.num_clones = 128
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

class cfg10:
    def __init__(self):
        self.name = "cfg10"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        self.func_pars = [1]
        self.num_clones = 256
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


class cfg11:
    def __init__(self):
        self.name = "cfg11"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        self.func_pars = [1]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 50
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=0.01, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.relu, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts]




class cfg12:
    def __init__(self):
        self.name = "cfg12"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 1 # internal batch size in the function
        inits = [tf.truncated_normal_initializer, tf.truncated_normal_initializer, tf.truncated_normal_initializer]
        trainables = [True,True,True]
        self.func_pars = [b_size,inits,trainables]
        self.num_clones = 1
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=0.01, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.relu, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts]


class cfg13:
    def __init__(self):
        self.name = "cfg13"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 1 # internal batch size in the function
        inits = [tf.truncated_normal_initializer, tf.truncated_normal_initializer, tf.truncated_normal_initializer]
        trainables = [True,True,True]
        self.func_pars = [b_size,inits,trainables]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=0.01, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.relu, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts]

class cfg14:
    def __init__(self):
        self.name = "cfg14"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 1 # internal batch size in the function
        inits = [tf.truncated_normal_initializer, tf.truncated_normal_initializer, tf.truncated_normal_initializer]
        trainables = [True,True,True]
        self.func_pars = [b_size,inits,trainables]
        self.num_clones = 16
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=0.01, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.relu, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts]

class cfg15:
    def __init__(self):
        self.name = "cfg15"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 1 # internal batch size in the function
        inits = [tf.truncated_normal_initializer, tf.truncated_normal_initializer, tf.truncated_normal_initializer]
        trainables = [True,True,True]
        self.func_pars = [b_size,inits,trainables]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=0.01, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.relu, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts]


class cfg16:
    def __init__(self):
        self.name = "cfg16"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 1 # internal batch size in the function
        inits = [tf.truncated_normal_initializer, tf.truncated_normal_initializer, tf.truncated_normal_initializer]
        trainables = [False,True,False]
        self.func_pars = [b_size,inits,trainables]
        self.num_clones = 1
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=0.01, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.relu, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts]


class cfg17:
    def __init__(self):
        self.name = "cfg17"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 1 # internal batch size in the function
        inits = [tf.truncated_normal_initializer, tf.truncated_normal_initializer, tf.truncated_normal_initializer]
        trainables = [False,True,False]
        self.func_pars = [b_size,inits,trainables]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=0.01, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.relu, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts]

class cfg18:
    def __init__(self):
        self.name = "cfg18"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 1 # internal batch size in the function
        inits = [tf.truncated_normal_initializer, tf.truncated_normal_initializer, tf.truncated_normal_initializer]
        trainables = [False,True,False]
        self.func_pars = [b_size,inits,trainables]
        self.num_clones = 16
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=0.01, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.relu, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts]

class cfg19:
    def __init__(self):
        self.name = "cfg19"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 1 # internal batch size in the function
        inits = [tf.truncated_normal_initializer, tf.truncated_normal_initializer, tf.truncated_normal_initializer]
        trainables = [False,True,False]
        self.func_pars = [b_size,inits,trainables]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=0.01, train_b=False).doit
        # fixme trainabe
        embedding_acts = [tf.nn.relu, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts]


class cfg20:
    def __init__(self):
        self.name = "cfg12"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 1 # internal batch size in the function
        inits = [tf.truncated_normal_initializer, tf.truncated_normal_initializer, tf.truncated_normal_initializer]
        trainables = [True,True,True]
        self.func_pars = [b_size,inits,trainables]
        self.num_clones = 1
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=True, init_b=0.01, train_b=True).doit # fixme trainabe
        embedding_acts = [tf.nn.relu, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts]


class cfg21:
    def __init__(self):
        self.name = "cfg21"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 1 # internal batch size in the function
        inits = [tf.truncated_normal_initializer, tf.truncated_normal_initializer, tf.truncated_normal_initializer]
        trainables = [True,True,True]
        self.func_pars = [b_size,inits,trainables]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=True, init_b=0.01, train_b=True).doit # fixme trainabe
        embedding_acts = [tf.nn.relu, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts]

class cfg22:
    def __init__(self):
        self.name = "cfg22"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 1 # internal batch size in the function
        inits = [tf.truncated_normal_initializer, tf.truncated_normal_initializer, tf.truncated_normal_initializer]
        trainables = [True,True,True]
        self.func_pars = [b_size,inits,trainables]
        self.num_clones = 16
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=True, init_b=0.01, train_b=True).doit # fixme trainabe
        embedding_acts = [tf.nn.relu, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts]

class cfg23:
    def __init__(self):
        self.name = "cfg23"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 1 # internal batch size in the function
        inits = [tf.truncated_normal_initializer, tf.truncated_normal_initializer, tf.truncated_normal_initializer]
        trainables = [True,True,True]
        self.func_pars = [b_size,inits,trainables]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=True, init_b=0.01, train_b=True).doit # fixme trainabe
        embedding_acts = [tf.nn.relu, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts]


class cfg24:
    def __init__(self):
        self.name = "cfg24"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 1 # internal batch size in the function
        inits = [tf.truncated_normal_initializer, tf.truncated_normal_initializer, tf.truncated_normal_initializer]
        trainables = [False,True,False]
        self.func_pars = [b_size,inits,trainables]
        self.num_clones = 1
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=True, init_b=0.01, train_b=True).doit # fixme trainabe
        embedding_acts = [tf.nn.relu, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts]


class cfg25:
    def __init__(self):
        self.name = "cfg25"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 1 # internal batch size in the function
        inits = [tf.truncated_normal_initializer, tf.truncated_normal_initializer, tf.truncated_normal_initializer]
        trainables = [False,True,False]
        self.func_pars = [b_size,inits,trainables]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=True, init_b=0.01, train_b=True).doit # fixme trainabe
        embedding_acts = [tf.nn.relu, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts]

class cfg26:
    def __init__(self):
        self.name = "cfg26"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 1 # internal batch size in the function
        inits = [tf.truncated_normal_initializer, tf.truncated_normal_initializer, tf.truncated_normal_initializer]
        trainables = [False,True,False]
        self.func_pars = [b_size,inits,trainables]
        self.num_clones = 16
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=True, init_b=0.01, train_b=True).doit # fixme trainabe
        embedding_acts = [tf.nn.relu, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts]

class cfg27:
    def __init__(self):
        self.name = "cfg27"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 1 # internal batch size in the function
        inits = [tf.truncated_normal_initializer, tf.truncated_normal_initializer, tf.truncated_normal_initializer]
        trainables = [False,True,False]
        self.func_pars = [b_size,inits,trainables]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=True, init_b=0.01, train_b=True).doit
        # fixme trainabe
        embedding_acts = [tf.nn.relu, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts]





class cfg28:
    def __init__(self):
        self.name = "cfg28"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256 # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20,20]
        self.func_pars = [b_size,inits,trainables,net_shapes]
        self.num_clones = 1
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        #scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "fit"]

class cfg29:
    def __init__(self):
        self.name = "cfg29"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256 # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20,20]
        self.func_pars = [b_size,inits,trainables,net_shapes]
        self.num_clones = 16
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        #scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "fit"]

class cfg30:
    def __init__(self):
        self.name = "cfg30"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256 # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20,20]
        self.func_pars = [b_size,inits,trainables,net_shapes]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        #scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "fit"]



class cfg31:
    def __init__(self):
        self.name = "cfg31"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 1
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        # scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg32:
    def __init__(self):
        self.name = "cfg32"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 16
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        # scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg33:
    def __init__(self):
        self.name = "cfg33"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        # scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]



class cfg34:
    def __init__(self):
        self.name = "cfg34"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 1
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        # scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg35:
    def __init__(self):
        self.name = "cfg35"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 16
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        # scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]

class cfg36:
    def __init__(self):
        self.name = "cfg36"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None,64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        # scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]



class cfg37:
    def __init__(self):
        self.name = "cfg37"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 1
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None,64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg38:
    def __init__(self):
        self.name = "cfg38"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 16
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None,64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]

class cfg39:
    def __init__(self):
        self.name = "cfg39"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]



class cfg40:
    def __init__(self):
        self.name = "cfg40"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 1
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        # scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg41:
    def __init__(self):
        self.name = "cfg41"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 16
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        # scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]

class cfg42:
    def __init__(self):
        self.name = "cfg42"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        # scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]



class cfg43:
    def __init__(self):
        self.name = "cfg43"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 1
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        # scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "fit"]


class cfg44:
    def __init__(self):
        self.name = "cfg44"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 16
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        # scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "fit"]

class cfg45:
    def __init__(self):
        self.name = "cfg45"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 100
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        # scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "fit"]


class cfg46:
    def __init__(self):
        self.name = "cfg46"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256 # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20,20]
        self.func_pars = [b_size,inits,trainables,net_shapes]
        self.num_clones = 1
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        #scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "fit"]

class cfg47:
    def __init__(self):
        self.name = "cfg47"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256 # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20,20]
        self.func_pars = [b_size,inits,trainables,net_shapes]
        self.num_clones = 16
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        #scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "fit"]

class cfg48:
    def __init__(self):
        self.name = "cfg48"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256 # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20,20]
        self.func_pars = [b_size,inits,trainables,net_shapes]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        #scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "fit"]



class cfg49:
    def __init__(self):
        self.name = "cfg49"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 1
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        # scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg50:
    def __init__(self):
        self.name = "cfg50"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 16
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        # scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg51:
    def __init__(self):
        self.name = "cfg51"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        # scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]



class cfg52:
    def __init__(self):
        self.name = "cfg52"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 1
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        # scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg53:
    def __init__(self):
        self.name = "cfg53"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 16
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        # scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]

class cfg54:
    def __init__(self):
        self.name = "cfg54"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None,64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        # scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]



class cfg55:
    def __init__(self):
        self.name = "cfg55"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 1
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None,64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg56:
    def __init__(self):
        self.name = "cfg56"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 16
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None,64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]

class cfg57:
    def __init__(self):
        self.name = "cfg57"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]



class cfg58:
    def __init__(self):
        self.name = "cfg58"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 1
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        # scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg59:
    def __init__(self):
        self.name = "cfg59"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 16
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        # scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]

class cfg60:
    def __init__(self):
        self.name = "cfg60"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        # scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]



class cfg61:
    def __init__(self):
        self.name = "cfg61"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 1
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        # scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "fit"]


class cfg62:
    def __init__(self):
        self.name = "cfg62"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 16
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        # scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "fit"]

class cfg63:
    def __init__(self):
        self.name = "cfg63"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(), tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        self.func_pars = [b_size, inits, trainables, net_shapes]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()
                           ]
        # scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "fit"]