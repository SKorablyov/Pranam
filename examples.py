import tensorflow as tf
import numpy as np
import sys,os
import utils,networks#,config
from config import *
import time
from pranam import PranamOptimizer


def test_pranam(cfg):
    # setup
    def _some_function():
        targets = tf.constant([1,2,3,4,5],dtype=tf.float32)
        guess = tf.Variable([5,4,3,2,1],dtype=tf.float32)
        cost = tf.reduce_mean((targets - guess)**2)
        return cost,[cost]

    # regular optimizer
    cost, adam_metrics = _some_function()
    adam_step = cfg.optimizer_params[0](*cfg.optimizer_params[1:]).minimize(cost)

    # same optimizer with parametrization
    pranam = PranamOptimizer(sess=cfg.sess, func=_some_function, func_pars=[], num_clones=cfg.num_clones,
                             optimizer_params=cfg.optimizer_params, batch_size=cfg.batch_size, embed_vars=None,
                             embedder_params=cfg.embedder_params)
    pranam_step, pranam_metrics = pranam.train_step()

    cfg.sess.run(tf.global_variables_initializer())
    for i in range(cfg.num_steps):
        _, _adam_cost = cfg.sess.run([adam_step, adam_metrics])
        print "adam_step:", i, "adam_cost:", _adam_cost
        _, _pranam_cost = cfg.sess.run([pranam_step, pranam_metrics])
        print "pranam_step:", i, "pranam_cost:", _pranam_cost


def schwefel_net_adam(cfg):
    cost, metrics = cfg.func(*cfg.func_pars)
    sess = tf.Session()
    #coord = tf.train.Coordinator()
    train_step = tf.train.GradientDescentOptimizer(learning_rate=cfg.learning_rate).minimize(cost)
    sess.run(tf.global_variables_initializer())
    #tf.train.start_queue_runners(sess, coord)
    for i in range(cfg.num_steps):
        _, _metrics = sess.run([train_step, metrics])
        _cost, _coords = _metrics
        print _cost
        print "step:", i, "mean_cost:",np.mean(_cost),"min_cost", np.min(_cost)


def schwefel_net_pranam(cfg):
    # initialize parameters to run
    #cfg.learning_rate = tf.placeholder(tf.float32)
    opt = PranamOptimizer(sess=cfg.sess, func=cfg.func, func_pars=cfg.func_pars, num_clones=cfg.num_clones,
                          optimizer_params=cfg.optimizer_params, batch_size=cfg.batch_size, embed_vars=cfg.embed_vars,
                          embedder_params=cfg.embedder_params)
    train_op, metrics = opt.train_step()
    #cost, metrics = cfg.func(*cfg.func_pars)
    sess = tf.Session()
    #coord = tf.train.Coordinator()
    #train_step = tf.train.GradientDescentOptimizer(learning_rate=cfg.learning_rate).minimize(cost)
    sess.run(tf.global_variables_initializer())
    #tf.train.start_queue_runners(sess, coord)
    for i in range(cfg.num_steps):
        _, _metrics = sess.run([train_op, metrics])
        _cost, _coords = _metrics
        print "step:",i,"mean_cost:",np.mean(_cost),"min_cost", np.min(_cost)#,# "min_coords", _coords[np.argmin(_cost)]


def mnist_fcnet_adam(cfg):
    # initialize parameters to run
    tr_loss, metrics = networks.mnist_fcnet(*cfg.func_pars)
    train_step = cfg.optimizer(cfg.learning_rate).minimize(tr_loss)
    sess = tf.Session()
    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess, coord)
    # run in the loop
    tm = utils.TrainingMonitor(cfg.name, cfg.out_path, n_ave=(5000//cfg.batch_size))
    for b_num in range(cfg.num_epochs * 60000 / cfg.batch_size):
        _, _metrics = sess.run([train_step, metrics])
        _metrics = {key: value for key, value in zip(["tr_loss", "tr_acc", "te_loss", "te_acc"], _metrics)}
        tm.add_many("_", _metrics, b_num)

        if b_num % 200 == 199:
            tm.check_save()

def mnist_fcnet_pranam(cfg):
    # initialize parameters to run
    opt = PranamOptimizer(sess=cfg.sess, func=cfg.func, func_pars=cfg.func_pars, num_clones=cfg.num_clones,
                          optimizer_params=cfg.optimizer_params, batch_size=cfg.batch_size,
                          embed_vars=cfg.embed_vars, embedder_params=cfg.embedder_params)
    train_op, metrics = opt.train_step()
    tr_loss_mean = tf.reduce_mean(metrics[0],axis=0)
    tr_acc_mean = tf.reduce_mean(metrics[1],axis=0)
    te_loss_mean = tf.reduce_mean(metrics[2],axis=0)
    te_acc_mean = tf.reduce_mean(metrics[3],axis=0)
    te_loss_min = tf.gather(metrics[2],tf.argmin(metrics[0]))
    te_acc_min = tf.gather(metrics[3],tf.argmin(metrics[0]))
    # summaries
    tf.summary.scalar("tr_loss_mean", tr_loss_mean)
    tf.summary.scalar("tr_acc_mean", tr_acc_mean)
    tf.summary.scalar("te_loss_mean", te_loss_mean)
    tf.summary.scalar("te_acc_mean", te_acc_mean)
    tf.summary.scalar("te_loss_min", te_loss_min)
    tf.summary.scalar("te_acc_min", te_acc_min)
    # initializers
    max_VRAM = tf.contrib.memory_stats.MaxBytesInUse()
    tf.summary.scalar("max_VRAM", max_VRAM)
    summary = tf.summary.merge_all()
    swriter = tf.summary.FileWriter(logdir=os.path.join(cfg.out_path, cfg.name))

    cfg.sess.run(tf.global_variables_initializer())  # fixme only initialize uninitialized variables
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=cfg.sess, coord=coord)
    # run in the loop
    tm = utils.TrainingMonitor("tm", os.path.join(cfg.out_path,cfg.name), n_ave=5000) # batch size is actually 1
    for i in range(cfg.num_epochs * 60000):
        _, _tr_loss_mean, _tr_acc_mean, _te_loss_mean, _te_acc_mean,_te_loss_min,_te_acc_min,_summary = cfg.sess.run(
            [train_op, tr_loss_mean, tr_acc_mean, te_loss_mean, te_acc_mean, te_loss_min, te_acc_min, summary])
        keys = ["tr_loss_mean", "tr_acc_mean", "te_loss_mean", "te_acc_mean", "te_loss_min", "te_acc_min"]
        values = [_tr_loss_mean, _tr_acc_mean, _te_loss_mean, _te_acc_mean, _te_loss_min, _te_acc_min]
        _metrics = {key: value for key, value in zip(keys, values)}
        tm.add_many("_", _metrics, i)

        print i

        if i % 50 == 19: # fixme temp
            tm.check_save()
            swriter.add_summary(_summary, i)

if __name__ == "__main__":
    # set up the config and folders
    #config_name = "cfg_t1"
    #example_name = "test_pranam"
    #config_name = "cfg94"
    #example_name = "mnist_fcnet_pranam"
    config_name = "cfg_s2"
    example_name = "schwefel_net_pranam"
    #config_name = "cfg_a3"
    #example_name = "mnist_fcnet_adam"

    if len(sys.argv) >= 2:
        config_name = sys.argv[1]
    if len(sys.argv) >= 3:
        example_name = sys.argv[2]
    cfg = eval(config_name)()
    if not os.path.exists(cfg.out_path): os.makedirs(cfg.out_path)
    eval(example_name)(cfg)