"""
# temporary file to experiment with writing an optimizer
# in this case optimizer will have to replicate the nodes to get something done

# optimize any network with a bigger fully-connected network
# and if that does not work, I will need variations
"""
import time
import tensorflow as tf
import numpy as np
import itertools,collections

# function_network(data=data, schema=schema)
# return loss, metrics

# pranam_optimizer(function_network=function_network,embed_shape)
#    loss,_ = function_network
#    trainable_variables = _get_gradient(loss) # build a list of trainable variables (could be a graph)
#    embedding = build_perceptron(embed_shape)

#    for i in range(embed_shape[0]):
#        loss, metric = replace_graph(function_network)
#  train_step = tf.train.AdamOptimizer(losses)

# return losses, metrics, train_step

# 1. I need to check if I have get_graph
# 2. I need to check how to replace a variable in the graph with some operations



def network(net_pars):
    dummy1 = tf.Variable(10)
    a = tf.get_variable("a",shape=[1,3,5,5], initializer=tf.initializers.random_uniform())
    b = tf.get_variable("b",shape=[], initializer=tf.initializers.random_uniform())
    c = tf.get_variable("c",shape=[],initializer=tf.initializers.random_uniform())
    d = tf.get_variable("d",shape=[],initializer=tf.initializers.random_uniform())
    z = (a + b) * (c + d)
    cost = (z - 7)**2
    dummy2 = tf.Variable(20)
    return cost, [tf.reduce_sum(cost)]


def rescale(x, a, b, c, d):
    """
    Rescales variable from [a, b] to [c, d]
    """
    return c + ((d - c) / (b - a)) * (x - a)


# def schwefel(x, xmin=-1, xmax=1):
#     """
#     https://www.sfu.ca/~ssurjano/schwef.html
#     """
#     x = rescale(x, xmin, xmax, -500, 500)
#     result = 418.9829 * tf.to_float(tf.shape(x)[1]) - tf.reduce_sum(tf.sin(tf.abs(x)**0.5) * x,axis=1)
#     return result

def schwefel(x, xmin=-1, xmax=1):
    """
    https://www.sfu.ca/~ssurjano/schwef.html
    """
    x = rescale(x, xmin, xmax, -500, 500)
    result = 418.9829 * tf.to_float(tf.shape(x)[0]) - tf.reduce_sum(tf.sin(tf.abs(x)**0.5) * x)
    return result


def schwefel_net(dim=10):
    guess = tf.get_variable(name="schnet_guess",shape=[dim],initializer=tf.random_uniform_initializer())
    cost = schwefel(guess)
    return cost,[cost]





class FCEmbedder:
    def __init__(self,sess,variables,gradients,embed_shape):
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
        embed_params = tf.get_variable("FCEmbed_0", #shape=[embed_shape[0], embed_shape[1]],
                                       initializer=np.asarray(np.random.normal(size=[embed_shape[0],embed_shape[1]]),np.float32))
                                      # fixme !!!!!!!tf.contrib.layers.xavier_initializer()
        top_layer = tf.nn.embedding_lookup(params=embed_params, ids=input)
        for i in range(1, len(embed_shape) - 1):
            w = tf.get_variable("FCEmbed_" + str(i), #shape=[embed_shape[i], embed_shape[i + 1]],
                                initializer=tf.random_uniform(shape=[embed_shape[i], embed_shape[i + 1]]))
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!fixme#tf.contrib.layers.xavier_initializer())
            top_layer = tf.matmul(top_layer, w)
            if i < (len(embed_shape) - 2):
                top_layer = tf.nn.relu(top_layer)
        # rescale with tg activation
        sess.run(tf.global_variables_initializer())  # FIXME: I need to initialize each of the weights separately !
        scaling = 1.7159 / tf.maximum(tf.abs(tf.reduce_max(top_layer)), tf.abs(tf.reduce_min(top_layer)))
        scaling_const = sess.run(scaling)
#        a_var = tf.get_variable("FCEmbed_a", dtype=tf.float32, shape=[], initializer=tf.constant_initializer(1.0))
#        b_var = tf.get_variable("FCEmbed_b", dtype=tf.float32, shape=[], initializer=tf.constant_initializer(0.001))
#        embedding_output = b_var * tf.nn.tanh(tf.multiply(scaling_const * top_layer, a_var))
        embedding_output = tf.nn.tanh(scaling_const * top_layer)

        # compute the gradient on embedding
        input_gradients = []
        for grads in gradients:
            input_gradients.append(tf.concat([tf.reshape(g,[-1]) for g in grads],axis=0))
        input_gradients = tf.stack(input_gradients,axis=0)
        embedding_loss = (embedding_output - tf.stop_gradient(embedding_output) + tf.stop_gradient(input_gradients))**2
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
                sl = [slices[var_num][0],slices[var_num][1]]
                embedded_var = tf.slice(embedding_output[clone_num,:],[sl[0]],[sl[1]])
                embedded_var = tf.reshape(embedded_var,varshapes[var_num])
                self.embedded_variables[clone_num].append(embedded_var)

    def compute_gradients(self):
        "compute and return state of embedding, and gradients on embedding"
        return self.embedded_variables, self.embedding_grads,self.embedding_vars


class GradientAccumulator:
    def __init__(self, optim, trainable_vars):
        self.optim = optim
        self.variable_names = np.array([v.name for v in trainable_vars])
        self.trainable_vars = trainable_vars
        self.gradients = [[] for v in trainable_vars]
        self.assign_vals = [None for v in trainable_vars]

    def add_gradients(self,gradients,variables):
        for i in range(len(gradients)):
            idx = np.where(self.variable_names == variables[i].name)[0][0]
            self.gradients[idx].append(gradients[i])

    def assign_vars(self,values,variables):
        for i in range(len(values)):
            idx = np.where(self.variable_names == variables[i].name)[0][0]
            self.assign_vals[idx] = values[i]

    def train_step(self,batch_size):
        # initialize gradient accumulator
        gradients = [] # flat gradients of a single tensor
        grad_accum = []
        for grad in self.gradients:
            if grad == []:
                gradients.append(None)
                grad_accum.append(None)
            else:
                grad = tf.add_n([tf.convert_to_tensor(g) for g in grad])
                gradients.append(grad)
                grad_accum.append(tf.Variable(tf.zeros_like(grad), trainable=False))

        # make a step counter to increment
        global_step = tf.Variable(1, name='global_step', trainable=False, dtype=tf.int64)
        incr_global_step = tf.assign(global_step, global_step + 1)
        remainder = tf.mod(incr_global_step, batch_size) < 1

        # create options to either merge or apply gradient depending on the step
        def _merge_grad(gradients=gradients,grad_accum=grad_accum):
            assign_ops = []
            for i in range(len(gradients)):
                if grad_accum[i] is not None:
                    assign_ops.append(tf.assign(grad_accum[i], grad_accum[i] + gradients[i]))
            with tf.control_dependencies(assign_ops):
                step_done = tf.constant(0)
            return step_done

        def _apply_grad(grad_accum=grad_accum, assign_vals=self.assign_vals, optim=self.optim,
                        trainable_vars=self.trainable_vars, b_size=batch_size):
            nvars = len(trainable_vars)
            # compute gradient averages
            grad_ave = []
            for i in range(len(grad_accum)):
                if grad_accum[i] is not None:
                    grad_ave.append(grad_accum[i] / b_size)
                else:
                    grad_ave.append(None)
            apply_grad = optim.apply_gradients(zip(grad_ave, trainable_vars))
            # flush gradient accumulator
            assign_zero = []
            with tf.control_dependencies([apply_grad]):
                for i in range(len(grad_accum)):
                    if grad_accum[i] is not None:
                        assign_zero.append(tf.assign(grad_accum[i], tf.zeros_like(grad_accum[i])))
            # assign variables that are hard-defined
            assignval_op = [tf.assign(trainable_vars[i],assign_vals[i]) for i in range(nvars) if assign_vals[i] != None]
            # empty option since grads run in the background
            with tf.control_dependencies(assign_zero + assignval_op):
                step_done = tf.constant(1)
            return step_done

        apply_gradient = tf.cond(remainder, true_fn=_apply_grad, false_fn=_merge_grad)
        return apply_gradient


class PranamOptimizer:
    def __init__(self, sess, func, func_pars, num_clones=16, optimizer=tf.train.AdamOptimizer,learning_rate=1e-5,
                 batch_size=100, embed_vars=None, embedder=FCEmbedder, embedder_pars=[[None,None]]):

        # call function num_clones times, and get a loss to optimize
        costs = []
        self.metrics = []
        for i in range(num_clones):
            with tf.variable_scope("clone_" + str(i)):
                cost, metric = func(*func_pars)
                costs.append(cost)
                self.metrics.append(metric)
        cost_mean = tf.reduce_mean(tf.stack(costs,axis=0))
        optim = optimizer(learning_rate=learning_rate) # todo pass pars

        # sort coupled and decoupled variables
        decoupled_grads = []
        decoupled_vars = []
        coupled_vars = [[] for i in range(num_clones)]
        coupled_grads = [[] for i in range(num_clones)]

        for grad, var in optim.compute_gradients(loss=cost_mean):
            clone_num = int(var.name.split("/")[0][6:])
            var_name = "/".join(var.name.split("/")[1:])

            if (grad is not None):
                if ((embed_vars is None) or (var_name in embed_vars)):
                    # couple variable
                    coupled_grads[clone_num].append(grad)
                    coupled_vars[clone_num].append(var)
                else:
                    # do not couple varaible
                    decoupled_grads.append(grad)
                    decoupled_vars.append(var)
            else:
                assert not (var_name in embed_vars), "no gradient available for" +str(var_name)

        if embed_vars is not None:
            for vars in coupled_vars:
                assert len(vars) == len(embed_vars), "could not embed one or more variables"
        else:
            for vars in coupled_vars:
                assert len(vars) == len(coupled_vars[0]), "broken embedding, trainable variables of a different length"

        # compute the update from the embedding
        embedding = embedder(sess,coupled_vars,coupled_grads,*embedder_pars)
        embedded_vars, embedding_grads, embedding_vars = embedding.compute_gradients()

        # prime the current optimizer with embedding_vars to create lr and momentum Variables
        fake_loss = tf.reduce_mean(tf.stack([tf.reduce_mean(v) for v in embedding_vars]))
        _,trainable_vars = zip(*optim.compute_gradients(fake_loss))

        # accumulate gradient
        ga = GradientAccumulator(optim,trainable_vars)
        ga.add_gradients(decoupled_grads,decoupled_vars)
        ga.add_gradients(embedding_grads,embedding_vars)
        embedded_vars = list(itertools.chain.from_iterable(embedded_vars)) # flatten [-1]
        coupled_vars = list(itertools.chain.from_iterable(coupled_vars))   # flatten [-1]
        ga.assign_vars(embedded_vars,coupled_vars)
        self.train_op = ga.train_step(batch_size)

    def train_step(self):
        metrics = [tf.stack(list(m),axis=0) for m in zip(*self.metrics)]
        return self.train_op,metrics


sess = tf.Session()
# test regular gradient convergence
#optimizer = PranamOptimizer(sess=sess,func=network,func_pars=[0],embed_vars=["a:0","b:0"],batch_size=5)

opt = PranamOptimizer(sess=sess,func=schwefel_net,num_clones=256,optimizer=tf.train.GradientDescentOptimizer,
                      func_pars=[10],embed_vars=None,batch_size=1,
                      embedder_pars=[[None,256,None]],learning_rate=4e-6)




train_op,metrics = opt.train_step()
cost_mean = tf.reduce_mean(metrics[0],axis=0)

sess.run(tf.global_variables_initializer()) # fixme only initialize not initialized variables


for i in range(1000):
    print sess.run([train_op,cost_mean])










# replace trainable variable b with embedding
# maybe, I could use same variable, but it could be simpler to play with gradient updates

# assign(a,embedding)
# gradients = stack([g1,g2,g3,g4,g5])

# the top network should try L2 loss to try to get 0 gradients everywhere
# how to send updates
# use another Adam with the same setup
# use L2_loss(gradients); compute gradient over the whole embedding; update the whole embedding


# [the gradient that arrives is a completely random thing???
# yes, initially
# later we are learning which things correlate, and which things anticorrelate, and more]

# now when the embedding has been updated, what do we do??
# 1. we do not need to update weights that are embedded at all
# 2. we need to update all other weights as usual

# special cases
# 0. The gradient on each of the weights is 0: we are done; embedding never updated

# 1. Embedding is initialized as Xavier, and weights of the network are initialized as that embedding = 6.860
# 2. Embedding is all 0s; there will be the gradient everywhere in the end of the day

# 3. Weights are initialized as Xavier, and the network is initialized as Xavier
# this is similar to initializing network to Xavier, and subtracting a constant from thing to make everything 0
# embedding = [[100,64],[64,32]] -> [100,32]

# we really don't care what the embedding outputs as we are gathering gradients; we need the last activation though
# the last activation will change the way gradients are distributed

# I think for faster saturation I need to connect two things
# two independent variables; one is (0.001 norm 0.002) and the top is (1000*1000) = (1,000,000)
# this may not update properly due to float point errors; I don't have that precision of 1e-6 at 1000
# I can't shift here!
# I can start with any embedding; and just train it to produce the distribution of the weights (and I think this is OK)

# solution: initialize network as anything; and initialize embedding as anything; train one to produce the other


def _accumulated_grad(b_size,cost,optimizer):
    """Auto accumulate grad and apply every Xts step"""
    # make make gradient accumulator
    optimizer = eval(optimizer)()
    grad, var = zip(*optimizer.compute_gradients(loss=cost))
    grad_accum = []
    for g in grad:
        if g is None:
            grad_accum.append(None)
        else:
            grad_accum.append(tf.Variable(tf.zeros_like(g), trainable=False))
    global_step = tf.Variable(1, name='global_step', trainable=False, dtype=tf.int64)
    incr_global_step = tf.assign(global_step, global_step + 1)
    remainder = tf.mod(incr_global_step, b_size) < 1

    def _merge_grad(grad=grad, grad_accum=grad_accum):
        # assert none
        assign_ops = []
        for i in range(len(grad)):
            if grad_accum[i] is not None:
                assign_ops.append(tf.assign(grad_accum[i], grad_accum[i] + grad[i]))
        with tf.control_dependencies(assign_ops):
            step_done = tf.constant(0)
        return step_done

    def _apply_grad(grad_accum=grad_accum, optimizer=optimizer, var=var, b_size=b_size):
        # compute gradient averages
        grad_ave = []
        for i in range(len(grad_accum)):
            if grad_accum[i] is not None:
                grad_ave.append(grad_accum[i] / b_size)
            else:
                grad_ave.append(None)
        apply_grad = optimizer.apply_gradients(zip(grad_ave, var))
        # flush gradient accumulator
        assign_zero = []
        with tf.control_dependencies([apply_grad]):
            for i in range(len(grad_accum)):
                if grad_accum[i] is not None:
                    assign_zero.append(tf.assign(grad_accum[i], tf.zeros_like(grad_accum[i])))
        # empty option since grads run in the background
        with tf.control_dependencies(assign_zero):
            step_done = tf.constant(1)
        return step_done
    apply_gradient = tf.cond(remainder, true_fn=_apply_grad, false_fn=_merge_grad)
    return apply_gradient




def _make_batch_v5(self, farg, opt_names=[], optimizers=[]):
    """
    Makes a batch with accumulator
    """
    net_feat, transit_par = self._single_ex_pipeline(*farg)  # compute network features for a single example
    num_opt = len(opt_names)
    train_ops = []
    # make summaries
    for key, value in net_feat.iteritems():
        if len(value.get_shape()) == 0:
            tf.summary.scalar(key, value)
    # make training options
    for i in range(num_opt):
        train_op = _accumulated_grad(b_size=self.b_size, cost=net_feat[opt_names[i]], optimizer=optimizers[i])
        train_ops.append(train_op)
    return net_feat, transit_par, train_ops


