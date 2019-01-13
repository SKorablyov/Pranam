"""
# temporary file to experiment with writing an optimizer
# in this case optimizer will have to replicate the nodes to get something done


# optimize any network with a bigger fully-connected network
# and if that does not work, I will need variations
"""
import time
import tensorflow as tf




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



a = tf.Variable(1.0)
b = tf.Variable(3.0)
c = a * b
cost = (c - 7)**2


embedding = tf.Variable(5.0) * tf.Variable(7.0)


optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(cost)

# get gradient and trainable variables
_, var = zip(*optimizer.compute_gradients(loss=cost))
print var[0]

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


# L2(gradients * embed) idiotic embed ==0 is a solution

# I can start with any embedding; and just train it to produce the distribution of the weights (and I think this is OK)





sess = tf.Session()
sess.run(tf.global_variables_initializer())


#print "grad:", grad
#print "var:", var





for i in range(10):
    print sess.run([cost,train_step])























#
def _accumulated_grad(b_size,cost,optimizer):
    """Auto accumulate grad and apply every X batch"""
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


