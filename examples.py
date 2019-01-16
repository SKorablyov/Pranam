import tensorflow as tf
import sys,os
import utils,networks
from config import *
import time


def mnist_fcnet_adam(cfg=cfg2()):
    # initialize parameters to run
    tr_loss, metrics = networks.mnist_fcnet(cfg.b_size)
    train_step = tf.train.AdamOptimizer().minimize(tr_loss)
    sess = tf.Session()
    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess, coord)
    # run in the loop
    tm = utils.TrainingMonitor(cfg.name, cfg.out_path, n_ave=50)
    for b_num in range(cfg.num_epochs * 60000 / cfg.b_size):
        _, _metrics = sess.run([train_step, metrics])
        _metrics = {key: value for key, value in zip(["tr_loss", "tr_acc", "te_loss", "te_acc"], _metrics)}
        tm.add_many("_", _metrics, b_num)

        if b_num % 200 == 199:
            tm.check_save()






if __name__ == "__main__":
    # set up the config and folders
    config_name = "cfg1"
    if len(sys.argv) >= 2:
        config_name = sys.argv[1]
    cfg = eval(config_name)()
    if not os.path.exists(cfg.out_path): os.makedirs(cfg.out_path)

    mnist_fcnet_adam()