import time
import tensorflow as tf
import numpy as np
import itertools,collections
from embedders import FCEmbedder


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
            adgrad_ops = []
            for i in range(len(gradients)):
                if grad_accum[i] is not None:
                    adgrad_ops.append(tf.assign(grad_accum[i], grad_accum[i] + gradients[i]))
            with tf.control_dependencies(adgrad_ops):
                step_done = tf.constant(0)
            return step_done

        def _merge_apply_grad(grad_accum=grad_accum, assign_vals=self.assign_vals, optim=self.optim,
                        trainable_vars=self.trainable_vars, b_size=batch_size):
            nvars = len(trainable_vars)

            # merge gradient from this batch
            adgrad_ops = []
            for i in range(len(gradients)):
                if grad_accum[i] is not None:
                    adgrad_ops.append(tf.assign(grad_accum[i], grad_accum[i] + gradients[i]))

            # compute gradient averages
            with tf.control_dependencies(adgrad_ops):
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

        apply_gradient = tf.cond(remainder, true_fn=_merge_apply_grad, false_fn=_merge_grad)
        return apply_gradient


class PranamOptimizer:
    def __init__(self, sess, func, func_pars, num_clones=16, optimizer=tf.train.AdamOptimizer,learning_rate=1e-5,
                 batch_size=100, embed_vars=None, embedder=FCEmbedder, embedder_pars=[[None,None]]):

        # call function num_clones times, and get a loss to optimize
        costs = []
        self.metrics = []
        for i in range(num_clones):
            with tf.variable_scope("clone_" + str(i)):

                #print func_pars
                #time.sleep(1000)

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
            var_name = str("/".join(var.name.split("/")[1:]))
            #print var_name,embed_vars

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


