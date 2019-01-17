import tensorflow as tf
import sys,os
import utils,networks#,config
from config import *
import time
from pranam import PranamOptimizer



def mnist_fcnet_adam(cfg):
    # initialize parameters to run
    tr_loss, metrics = networks.mnist_fcnet(cfg.b_size)
    train_step = tf.train.AdamOptimizer().minimize(tr_loss)
    sess = tf.Session()
    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess, coord)
    # run in the loop
    tm = utils.TrainingMonitor(cfg.name, cfg.out_path, n_ave=(5000//cfg.b_size))
    for b_num in range(cfg.num_epochs * 60000 / cfg.b_size):
        _, _metrics = sess.run([train_step, metrics])
        _metrics = {key: value for key, value in zip(["tr_loss", "tr_acc", "te_loss", "te_acc"], _metrics)}
        tm.add_many("_", _metrics, b_num)

        if b_num % 200 == 199:
            tm.check_save()


def mnist_fcnet_pranam(cfg):
    # initialize parameters to run
    opt = PranamOptimizer(sess=cfg.sess, func=cfg.func, func_pars=cfg.func_pars, num_clones=cfg.num_clones,
                          optimizer=cfg.optimizer, learning_rate=cfg.learning_rate, batch_size=cfg.batch_size,
                          embed_vars=cfg.embed_vars, embedder=cfg.embedder, embedder_pars=cfg.embedder_pars)
    train_op, metrics = opt.train_step()
    tr_loss = tf.reduce_mean(metrics[0], axis=0)
    tr_acc = tf.reduce_mean(metrics[1], axis=0)
    te_loss = tf.reduce_mean(metrics[2], axis=0)
    te_acc = tf.reduce_mean(metrics[3], axis=0)
    cfg.sess.run(tf.global_variables_initializer())  # fixme only initialize uninitialized variables
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=cfg.sess, coord=coord)
    # run in the loop
    tm = utils.TrainingMonitor(cfg.name, cfg.out_path, n_ave=5000) # batch size is actually 1

    for i in range(cfg.num_epochs * 60000):
        _, _tr_loss, _tr_acc, _te_loss, _te_acc = cfg.sess.run([train_op, tr_loss, tr_acc, te_loss, te_acc])
        keys = ["tr_loss", "tr_acc", "te_loss", "te_acc"]
        values = [_tr_loss, _tr_acc, _te_loss, _te_acc]
        _metrics = {key: value for key, value in zip(keys,values)}
        tm.add_many("_", _metrics, i)

        if i % 2000 == 1990:
            tm.check_save()

if __name__ == "__main__":
    # set up the config and folders
    config_name = "cfg20"
    example_name = "mnist_fcnet_pranam"
    if len(sys.argv) >= 2:
        config_name = sys.argv[1]
    if len(sys.argv) >= 3:
        example_name = sys.argv[2]
    cfg = eval(config_name)()
    if not os.path.exists(cfg.out_path): os.makedirs(cfg.out_path)
    eval(example_name)(cfg)
