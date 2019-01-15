import time, os, sys, socket
import tensorflow as tf
import numpy as np
import config as conf
from input import read_dataset
from input import generate_dataset
from networks import net3_embed



def _try_params(n_iterations, batch_size, fun_shape, em_shape, db_path, lr, optimizer, scheduler, net3, tl,counter):
    "try some parameters, report testing accuracy with square loss"
    # read data
    x_train, y_train, x_test, y_test = read_dataset(db_path, batch_size)
    # initialize training/testing graph
    # initialize session
    sess = tf.Session()
    coord = tf.train.Coordinator()
    _, yhat_train, X = eval(net3)(X=x_train, fun_shape=fun_shape, em_shape=em_shape, sess=sess, coord=coord, tl=tl)
    #find accuracy om train data
    y_ = tf.expand_dims(y_train, 1)
    y__=y_
    for i in range(em_shape[0]-1):

        y__=tf.concat([y__,y_],axis=1)
    yhat_predicted = tf.nn.softmax(yhat_train)

    correct_prediction = tf.equal(tf.argmax(yhat_predicted, 2), tf.argmax(y__, 2))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #find loss
    train_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=yhat_train, labels=y__)

    #find accuracy on test data
    l0 = tf.expand_dims(x_test, 1)
    W1 = X[0]
    l1 = tf.reduce_sum(tf.expand_dims(l0, 3) * W1, axis=2)
    l1_act = tf.nn.relu(l1)
    W2 = X[1]
    l2 = tf.reduce_sum(tf.expand_dims(l1_act, 3) * W2, axis=2)
    l2_act = tf.nn.relu(l2)
    W3 = X[2]
    yhat_train = tf.reduce_sum(tf.expand_dims(l2_act, 3) * W3, axis=2)
    y_ = tf.expand_dims(y_test, 1)
    y__=y_

    for i in range(em_shape[0]-1):

        y__=tf.concat([y__,y_],axis=1)

    yhat=tf.nn.softmax(yhat_train)

    correct_test_prediction = tf.equal(tf.argmax(yhat, 2), tf.argmax(y__, 2))
    test_acc_ = tf.reduce_mean(tf.cast(correct_test_prediction, tf.float32))

    lr_current = tf.placeholder(tf.float32)
    train_step = eval(optimizer)(learning_rate=lr_current).minimize(train_loss)

    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess, coord)
    _train_losses = []
    accuracy_container = []
    test_acc_container = []

    mean_accuracy = 0

    for i in range(n_iterations):

        start = time.time()

        _train_loss, accuracy ,t_accuracy,_ = sess.run([train_loss, acc,test_acc_, train_step], feed_dict={lr_current: lr})
        mean_accuracy += accuracy

        _train_losses.append(_train_loss)
        accuracy_container.append(accuracy)
        test_acc_container.append(t_accuracy)

        if scheduler == "none":
            pass
        elif scheduler == "dlp":
            # scheduler the step size
            if i % 2000 == 1999:
                if np.mean(_train_losses[-1000:]) >= np.mean(_train_losses[-2000:-1000]):
                    lr = lr * 0.5
        else:
            machine = socket.gethostname()
            if machine != "viacheslav-HP-Pavilion-Notebook":
                raise ValueError("unknown scheduler")

            else:
                pass
            if i % 100 == 0:
                # print "argmin of the train loss", _y_diff[_train_loss.argmin()],
                print("step:", i, "mean_loss", np.mean(_train_loss), "min_loss", np.min(_train_loss), )
                # print "y_train:", np.mean(_y_train), np.var(_y_train),_y_train.shape,
                # print "y_hat:", np.mean(_yhat_train),np.var(_yhat_train), _yhat_train.shape,
                print("mean accuracy=", mean_accuracy, "%")
                print("lr", lr)
                print("exps:", cfg.batch_size / (time.time() - start))
                mean_accuracy = 0

        # history
        if i % 1000 == 1:
            _train_loss, = sess.run([train_loss], feed_dict={lr_current: lr})

    sess.close()
    tf.reset_default_graph()
    np.savetxt(os.path.join(cfg.out_path, config_name + "_aam_train_accuracy_" + str(lr)+str(counter)), accuracy_container)
    np.savetxt(os.path.join(cfg.out_path, config_name + "_aam_test_accuracy_" + str(lr) + str(counter)),test_acc_container)

    sel_point = np.mean(_train_losses[-200:-100], axis=0).argmin()
    minmean_loss = np.mean(_train_losses[:-100], axis=0)[sel_point]
    loss_hist = np.asarray(_train_losses)[:, sel_point]

    return minmean_loss, loss_hist


def try_params(n_iterations, batch_size, fun_shape, em_shape, db_path, lr, optimizer, scheduler, net3, tl,counter):
    train_cost, train_cost_hist = tf.py_func(_try_params,
                                             [n_iterations, batch_size, fun_shape, em_shape, db_path, lr, optimizer,
                                              scheduler, net3, tl,counter],
                                             [tf.float32, tf.float32])
    sess = tf.Session()
    _train_cost, _train_cost_hist = sess.run([train_cost, train_cost_hist])
    print(_train_cost_hist)
    sess.close()
    return _train_cost, _train_cost_hist


if __name__ == "__main__":
    # set up the config and folders
    config_name = "cfg_mnist_2"
    if len(sys.argv) >= 2:
        config_name = sys.argv[1]

    cfg = eval("conf." + config_name)
    if not os.path.exists(cfg.out_path): os.makedirs(cfg.out_path)
    counter = 0
    for lr in cfg.lrs:
        counter=counter+1
        train_costs = []
        train_cost_hists = []
        for i in range(cfg.n_runs):
            # generate dataset
            generate_dataset(cfg.db_path, cfg.genf_shape, cfg.train_samples, cfg.test_samples, noise=cfg.noise)
            # train_cost,train_cost_hist = try_params(1000,cfg.batch_size,[64,32,1],cfg.db_path,cfg.test_samples, lr=lr)
            train_cost, train_cost_hist = try_params(n_iterations=cfg.n_iterations,
                                                     batch_size=cfg.batch_size,
                                                     fun_shape=cfg.fun_shape,
                                                     em_shape=cfg.em_shape,
                                                     db_path=cfg.db_path, lr=lr, optimizer=cfg.optimizer,
                                                     scheduler=cfg.scheduler, tl=cfg.training_layers,
                                                     net3="net3_embed",counter=counter)






