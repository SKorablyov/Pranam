import tensorflow as tf
import networks,embedders,utils


class cfg_t1:
    # tests simple gradient descent
    def __init__(self):
        self.name = "cfg_t1"
        self.optimizer_params = [tf.train.GradientDescentOptimizer, 0.01]
        self.out_path = "./results"
        self.sess = tf.Session()
        self.num_steps = 1000
        self.num_clones = 1
        self.batch_size = 1

        # embedding pars
        self.embed_vars = ["mnist_fcnet/fake_w3:0"]
        embedder = embedders.FCEmbedder
        embedding_shape = [None, None]
        def constant_init(a,partition_info,dtype):
            return tf.constant([[5, 4, 3, 2, 1]], dtype=dtype)
        embedding_inits = [constant_init]
        def no_act(x):
            return x
        embedding_acts = [no_act]
        self.embedder_params = [embedder, embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg_s1:
    def __init__(self):
        "network for schwefel -- needs fixes since runs out of range"
        self.name = "cfg1"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.schwefel_net
        self.func_pars = [10]
        self.num_clones = 250
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5 #4e-6
        self.batch_size = 1
        self.num_steps = 10000
        # embedding pars
        self.embed_vars = None
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 250, None]
        embedding_inits = [tf.random_normal_initializer,tf.random_uniform_initializer,tf.random_uniform_initializer]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit
        embedding_acts = [tf.nn.relu,scaled_tanh]
        self.embedder_pars = [embedding_shape,embedding_inits,embedding_acts, None]


class cfg_s2:
    def __init__(self):
        "network for schwefel -- needs fixes since runs out of range"
        self.name = "cfg1"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.schwefel_net
        self.func_pars = [10]
        self.num_clones = 100
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-4 #4e-6
        self.batch_size = 1
        self.num_steps = 20000
        # embedding pars
        self.embed_vars = None
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 250, None]
        embedding_inits = [tf.initializers.variance_scaling(scale=20.0, distribution="uniform"),
                           tf.initializers.variance_scaling(scale=20.0, distribution="uniform")
                           ]
        embedding_acts = [tf.nn.relu,tf.nn.tanh]
        self.embedder_pars = [embedding_shape,embedding_inits,embedding_acts, None]


class cfg_s3:
    def __init__(self):
        "network for schwefel -- needs fixes since runs out of range"
        self.name = "cfg1"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.schwefel_net
        self.func_pars = [10]
        self.num_clones = 100
        self.optimizer_params = [tf.train.GradientDescentOptimizer, 1e-3]#4e-6
        self.batch_size = 1
        self.num_steps = 20000
        # embedding pars
        self.embed_vars = None
        embedder = embedders.FCEmbedder
        embedding_shape = [None, 250, None]
        embedding_inits = [tf.initializers.variance_scaling(scale=10.0, distribution="uniform"),
                           tf.initializers.variance_scaling(scale=10.0, distribution="uniform")
                           ]
        embedding_acts = [tf.nn.relu,tf.nn.softsign]
        self.embedder_params = [embedder,embedding_shape,embedding_inits,embedding_acts, None]


class cfg_s4:
    def __init__(self):
        "network for schwefel -- needs fixes since runs out of range"
        self.name = "cfg1"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.schwefel_net
        self.func_pars = [400]
        self.num_clones = 100
        self.optimizer_params = [tf.train.GradientDescentOptimizer, 1e-3]#4e-6
        self.batch_size = 1
        self.num_steps = 20000
        # embedding pars
        self.embed_vars = None
        embedder = embedders.FCEmbedder
        embedding_shape = [None, 250, None]
        embedding_inits = [tf.initializers.variance_scaling(scale=10.0, distribution="uniform"),
                           tf.initializers.variance_scaling(scale=10.0, distribution="uniform")
                           ]
        embedding_acts = [tf.nn.relu,tf.nn.softsign]
        self.embedder_params = [embedder,embedding_shape,embedding_inits,embedding_acts, None]


class cfg_a2:
    def __init__(self):
        self.name = "cfg2"
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 0.001
        self.out_path = "./results"
        self.sess = tf.Session()
        self.batch_size = 256  # internal batch size in the function
        self.num_epochs = 50
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.softsign, tf.nn.softsign]
        self.func_pars = [self.batch_size, inits, trainables, net_shapes, acts]


class cfg_a3:
    def __init__(self):
        self.name = "cfg2"
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 0.001
        self.out_path = "./results"
        self.sess = tf.Session()
        self.batch_size = 256  # internal batch size in the function
        self.num_epochs = 50
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.relu, tf.nn.relu]
        self.func_pars = [self.batch_size, inits, trainables, net_shapes, acts]

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
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
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
    # works ok
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
    # does not work at all
    def __init__(self):
        self.name = "cfg29"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256 # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        net_shapes = [20, 20]
        trainables = [True, True, True]
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
    # ?
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
    # works well
    def __init__(self):
        self.name = "cfg31"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
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
    # barely works by far worse compared to 31
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
    # dead
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
    #
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
    # 53 and nowhere near the saturation
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
    # 81 and rising
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
    # smallest signs of life at 0.15 of acc
    def __init__(self):
        self.name = "cfg50"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
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
                           tf.contrib.layers.xavier_initializer()]
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
    # 62 and rising
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
    # 86 and rising
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
    # smallest signs of life at 0.12
    def __init__(self):
        self.name = "cfg57"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
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
    # acc 0.8 and rising
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
    # sumewhat sluggish uprisze at 30
    def __init__(self):
        self.name = "cfg61"
        self.out_path = "./results"
        "FC network on mnist"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
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


class cfg64:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg64"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.relu,tf.nn.relu]
        self.func_pars = [b_size, inits, trainables, net_shapes,acts]
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
                           tf.contrib.layers.xavier_initializer()]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg65:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg65"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.relu,tf.nn.relu]
        self.func_pars = [b_size, inits, trainables, net_shapes,acts]
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
                           tf.contrib.layers.xavier_initializer()]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.softsign, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg66:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg66"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.relu, tf.nn.relu]
        self.func_pars = [b_size, inits, trainables, net_shapes, acts]
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
                           tf.contrib.layers.xavier_initializer()]
        #scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.softsign, tf.nn.softsign]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg67:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg67"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.softsign, tf.nn.softsign]
        self.func_pars = [b_size, inits, trainables, net_shapes, acts]
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
                           tf.contrib.layers.xavier_initializer()]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.softsign, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg68:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg68"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.softsign, tf.nn.softsign]
        self.func_pars = [b_size, inits, trainables, net_shapes, acts]
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
                           tf.contrib.layers.xavier_initializer()]
        embedding_acts = [tf.nn.softsign, tf.nn.softsign]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg69:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg69"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.softsign, tf.nn.softsign]
        self.func_pars = [b_size, inits, trainables, net_shapes, acts]
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
                           tf.contrib.layers.xavier_initializer()]
        embedding_acts = [tf.nn.softsign, tf.nn.softsign]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg70:
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg70"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.softsign, tf.nn.softsign]
        self.func_pars = [b_size, inits, trainables, net_shapes, acts]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.initializers.variance_scaling(scale=20.0, distribution="uniform"),
                           tf.initializers.variance_scaling(scale=20.0, distribution="uniform")]
        embedding_acts = [tf.nn.relu, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg71:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg71"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.softsign, tf.nn.softsign]
        self.func_pars = [b_size, inits, trainables, net_shapes, acts]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.initializers.variance_scaling(scale=20.0, distribution="uniform"),
                           tf.initializers.variance_scaling(scale=20.0, distribution="uniform")]
        embedding_acts = [tf.nn.relu, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg72:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg72"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.relu, tf.nn.relu]
        self.func_pars = [b_size, inits, trainables, net_shapes, acts]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.initializers.variance_scaling(scale=20.0, distribution="uniform"),
                           tf.initializers.variance_scaling(scale=20.0, distribution="uniform")]
        embedding_acts = [tf.nn.relu, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg73:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg73"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.softsign, tf.nn.softsign]
        self.func_pars = [b_size, inits, trainables, net_shapes, acts]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 128, None]
        embedding_inits = [tf.initializers.variance_scaling(scale=20.0, distribution="uniform"),
                           tf.initializers.variance_scaling(scale=20.0, distribution="uniform")]
        embedding_acts = [tf.nn.relu, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg74:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg74"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.relu, tf.nn.relu]
        self.func_pars = [b_size, inits, trainables, net_shapes, acts]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 128, None]
        embedding_inits = [tf.initializers.variance_scaling(scale=20.0, distribution="uniform"),
                           tf.initializers.variance_scaling(scale=20.0, distribution="uniform")]
        embedding_acts = [tf.nn.relu, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg75:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg75"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.softsign, tf.nn.softsign]
        self.func_pars = [b_size, inits, trainables, net_shapes, acts]
        self.num_clones = 32
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.initializers.variance_scaling(scale=20.0, distribution="uniform"),
                           tf.initializers.variance_scaling(scale=20.0, distribution="uniform")]
        embedding_acts = [tf.nn.relu, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg76:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg76"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.relu, tf.nn.relu]
        self.func_pars = [b_size, inits, trainables, net_shapes, acts]
        self.num_clones = 32
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.initializers.variance_scaling(scale=20.0, distribution="uniform"),
                           tf.initializers.variance_scaling(scale=20.0, distribution="uniform")]
        embedding_acts = [tf.nn.relu, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg77:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg77"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.softsign, tf.nn.softsign]
        self.func_pars = [b_size, inits, trainables, net_shapes, acts]
        self.num_clones = 128
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 256, None]
        embedding_inits = [tf.initializers.variance_scaling(scale=20.0, distribution="uniform"),
                           tf.initializers.variance_scaling(scale=20.0, distribution="uniform")]
        embedding_acts = [tf.nn.relu, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg78:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg78"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.relu, tf.nn.relu]
        self.func_pars = [b_size, inits, trainables, net_shapes, acts]
        self.num_clones = 128
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 256, None]
        embedding_inits = [tf.initializers.variance_scaling(scale=20.0, distribution="uniform"),
                           tf.initializers.variance_scaling(scale=20.0, distribution="uniform")]
        embedding_acts = [tf.nn.relu, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg79:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg79"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.relu,tf.nn.relu]
        self.func_pars = [b_size, inits, trainables, net_shapes,acts]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-3
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.tanh, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg80:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg80"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.relu,tf.nn.relu]
        self.func_pars = [b_size, inits, trainables, net_shapes,acts]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-3
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.softsign, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg81:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg81"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.relu, tf.nn.relu]
        self.func_pars = [b_size, inits, trainables, net_shapes, acts]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-3
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()]
        #scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.softsign, tf.nn.softsign]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg82:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg82"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.softsign, tf.nn.softsign]
        self.func_pars = [b_size, inits, trainables, net_shapes, acts]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-3
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()]
        scaled_tanh = embedders.ScaledTanh("sctanh1", self.sess, init_a=1.0, train_a=False, init_b=1.0, train_b=False).doit # fixme trainabe
        embedding_acts = [tf.nn.softsign, scaled_tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg83:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg83"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.softsign, tf.nn.softsign]
        self.func_pars = [b_size, inits, trainables, net_shapes, acts]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-3
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()]
        embedding_acts = [tf.nn.softsign, tf.nn.softsign]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg84:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg84"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.softsign, tf.nn.softsign]
        self.func_pars = [b_size, inits, trainables, net_shapes, acts]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-3
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.contrib.layers.xavier_initializer(),
                           tf.contrib.layers.xavier_initializer()]
        embedding_acts = [tf.nn.softsign, tf.nn.softsign]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg85:
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg85"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.softsign, tf.nn.softsign]
        self.func_pars = [b_size, inits, trainables, net_shapes, acts]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-3
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.initializers.variance_scaling(scale=20.0, distribution="uniform"),
                           tf.initializers.variance_scaling(scale=20.0, distribution="uniform")]
        embedding_acts = [tf.nn.relu, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg86:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg86"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.softsign, tf.nn.softsign]
        self.func_pars = [b_size, inits, trainables, net_shapes, acts]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-3
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.initializers.variance_scaling(scale=20.0, distribution="uniform"),
                           tf.initializers.variance_scaling(scale=20.0, distribution="uniform")]
        embedding_acts = [tf.nn.relu, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg87:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg87"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.relu, tf.nn.relu]
        self.func_pars = [b_size, inits, trainables, net_shapes, acts]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-3
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.initializers.variance_scaling(scale=20.0, distribution="uniform"),
                           tf.initializers.variance_scaling(scale=20.0, distribution="uniform")]
        embedding_acts = [tf.nn.relu, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg88:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg88"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.softsign, tf.nn.softsign]
        self.func_pars = [b_size, inits, trainables, net_shapes, acts]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-3
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 128, None]
        embedding_inits = [tf.initializers.variance_scaling(scale=20.0, distribution="uniform"),
                           tf.initializers.variance_scaling(scale=20.0, distribution="uniform")]
        embedding_acts = [tf.nn.relu, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg89:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg89"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.relu, tf.nn.relu]
        self.func_pars = [b_size, inits, trainables, net_shapes, acts]
        self.num_clones = 64
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-3
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 128, None]
        embedding_inits = [tf.initializers.variance_scaling(scale=20.0, distribution="uniform"),
                           tf.initializers.variance_scaling(scale=20.0, distribution="uniform")]
        embedding_acts = [tf.nn.relu, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg90:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg90"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.softsign, tf.nn.softsign]
        self.func_pars = [b_size, inits, trainables, net_shapes, acts]
        self.num_clones = 32
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-3
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.initializers.variance_scaling(scale=20.0, distribution="uniform"),
                           tf.initializers.variance_scaling(scale=20.0, distribution="uniform")]
        embedding_acts = [tf.nn.relu, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg91:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg91"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.relu, tf.nn.relu]
        self.func_pars = [b_size, inits, trainables, net_shapes, acts]
        self.num_clones = 32
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-3
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 64, None]
        embedding_inits = [tf.initializers.variance_scaling(scale=20.0, distribution="uniform"),
                           tf.initializers.variance_scaling(scale=20.0, distribution="uniform")]
        embedding_acts = [tf.nn.relu, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg92:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg92"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.softsign, tf.nn.softsign]
        self.func_pars = [b_size, inits, trainables, net_shapes, acts]
        self.num_clones = 128
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-3
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 256, None]
        embedding_inits = [tf.initializers.variance_scaling(scale=20.0, distribution="uniform"),
                           tf.initializers.variance_scaling(scale=20.0, distribution="uniform")]
        embedding_acts = [tf.nn.relu, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg93:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg93"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.relu, tf.nn.relu]
        self.func_pars = [b_size, inits, trainables, net_shapes, acts]
        self.num_clones = 128
        self.optimizer = tf.train.GradientDescentOptimizer
        self.learning_rate = 1e-3
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/w2:0"]
        self.embedder = embedders.FCEmbedder
        embedding_shape = [None, 256, None]
        embedding_inits = [tf.initializers.variance_scaling(scale=20.0, distribution="uniform"),
                           tf.initializers.variance_scaling(scale=20.0, distribution="uniform")]
        embedding_acts = [tf.nn.relu, tf.nn.tanh]
        self.embedder_pars = [embedding_shape, embedding_inits, embedding_acts, "rescale"]


class cfg94:
    # should be 0.12 and go
    "FC network on mnist"
    def __init__(self):
        self.name = "cfg94"
        self.out_path = "./results"
        self.sess = tf.Session()
        # network and opt pars
        self.func = networks.mnist_fcnet
        b_size = 256  # internal batch size in the function
        inits = [tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer(),
                 tf.contrib.layers.xavier_initializer()]
        trainables = [True, True, True]
        net_shapes = [20, 20]
        acts = [tf.nn.relu, tf.nn.relu]
        self.func_pars = [b_size, inits, trainables, net_shapes, acts]
        self.num_clones = 1
        self.optimizer_params = [tf.train.GradientDescentOptimizer, 1e-3]
        self.batch_size = 1
        self.num_epochs = 100
        # embedding pars
        self.embed_vars = ["mnist_fcnet/fake_w3:0"]
        embedder = embedders.FCEmbedder
        embedding_shape = [None, None]
        embedding_inits = [tf.initializers.variance_scaling(scale=20.0, distribution="uniform"),
                           tf.initializers.variance_scaling(scale=20.0, distribution="uniform")]
        def no_act(x):
            return x
        embedding_acts = [no_act]
        self.embedder_params = [embedder, embedding_shape, embedding_inits, embedding_acts, "rescale"]