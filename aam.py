import time, os, sys, socket
import tensorflow as tf
import numpy as np
import config_2vs3l_emb as conf
from keras.datasets import mnist  # subroutines for fetching the MNIST dataset
from keras.utils import np_utils  # utilities for one-hot encoding of ground truth values


def hierarchical_compositional(genf_shape, n_samples, W=None, noise=None):
    "generate a bunch of samples, the activation function is frozen as relu"
    X = np.matrix(np.asarray(np.random.uniform(size=[n_samples, genf_shape[0]])))
    lastX = X
    # initialize
    if W is None:
        W = []
        for i in np.arange(len(genf_shape) - 1) + 1:
            w = np.matrix(np.random.uniform(low=-2, high=2, size=[genf_shape[i - 1], genf_shape[i]]))
            W.append(w)
    # apply weights, get Ys
    for i in np.arange(len(genf_shape) - 1):
        print("layer", i, "incoming shape", lastX.shape)
        # print "test shape", lastX.shape,W[i].shape, (lastX * W[i]).shape, np.maximum(lastX * W[i],0).shape
        lastX = np.maximum(np.asarray(lastX * W[i]), 0)
    Y = lastX
    if noise is not None:
        Y = Y + np.random.uniform(size=Y.shape, low=-0.5 * noise, high=0.5 * noise)
    return X, Y, W


def generate_dataset(out_path, f_shape, train_samples, test_samples, noise):
    if not os.path.exists(out_path): os.makedirs(out_path)
    op = str(out_path)
    op = op[-5:]

    if op == "mnist":
        num_train = 60000  # there are 60000 training examples in MNIST
        num_test = 10000  # there are 10000 test examples in MNIST

        height, width, depth = 28, 28, 1  # MNIST images are 28x28 and greyscale
        num_classes = 10  # there are 10 classes (1 per digit)

        (X_train, y_train), (X_test, y_test) = mnist.load_data()  # fetch MNIST data

        X_train = X_train.reshape(num_train, height * width)  # Flatten data to 1D
        X_test = X_test.reshape(num_test, height * width)  # Flatten data to 1D
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        #X_train /= 255  # Normalise data to [0, 1] range
        #X_test /= 255  # Normalise data to [0, 1] range
        Y_train = np_utils.to_categorical(y_train, num_classes)  # One-hot encode the labels
        Y_test = np_utils.to_categorical(y_test, num_classes)  # One-hot encode the labels
        print("data has been saved")
        np.save(os.path.join(out_path, "X_test"), np.asarray(X_test, np.float32))
        np.save(os.path.join(out_path, "Y_test"), np.asarray(Y_test, np.float32))
        np.save(os.path.join(out_path, "X_train"), np.asarray(X_train, np.float32))
        np.save(os.path.join(out_path, "Y_train"), np.asarray(Y_train, np.float32))

    else:
        # generate training set
        X_train, Y_train, W = hierarchical_compositional(f_shape, n_samples=train_samples, noise=noise)
        np.save(os.path.join(out_path, "X_train"), np.asarray(X_train, np.float32))
        np.save(os.path.join(out_path, "Y_train"), np.asarray(Y_train, np.float32))
        # generate testing set
        X_test, Y_test, _ = hierarchical_compositional(f_shape, n_samples=test_samples, W=W, noise=noise)
        np.save(os.path.join(out_path, "X_test"), np.asarray(X_test, np.float32))
        np.save(os.path.join(out_path, "Y_test"), np.asarray(Y_test, np.float32))
        print("data has been saved")


def read_dataset(db_path, batch_size):
    # load dataset
    machine = socket.gethostname()
    if machine == "viacheslav-HP-Pavilion-Notebook":

        path = str(db_path)
        path = path[2:]
        path = path[:-1]
        X_train = np.load(os.path.join(path, "X_train.npy"))
        Y_train = np.load(os.path.join(path, "Y_train.npy"))
        X_test = np.load(os.path.join(path, "X_test.npy"))
        Y_test = np.load(os.path.join(path, "Y_test.npy"))

    else:
        X_train = np.load(os.path.join(db_path, "X_train.npy"))
        Y_train = np.load(os.path.join(db_path, "Y_train.npy"))
        X_test = np.load(os.path.join(db_path, "X_test.npy"))
        Y_test = np.load(os.path.join(db_path, "Y_test.npy"))

    print("loaded dataset of shapes:", X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    # make batches
    x_train, y_train = tf.train.slice_input_producer([X_train, Y_train])
    x_train, y_train = tf.train.batch([x_train, y_train], batch_size)
    x_test, y_test = tf.train.slice_input_producer([X_test, Y_test])
    x_test, y_test = tf.train.batch([x_test, y_test], batch_size)
    return x_train, y_train, x_test, y_test


def rescale(x, a, b, c, d):
    """
    Rescales variable from [a, b] to [c, d]
    """
    return c + ((d - c) / (b - a)) * (x - a)


def schwefel(x, xmin=-1, xmax=1):
    """
    https://www.sfu.ca/~ssurjano/schwef.html
    """
    x = rescale(x, xmin, xmax, -500, 500)
    # result = x.abs().sqrt().sin().mul(x).sum(1).mul(-1).add(418.9829 * x.size(1))
    result = 418.9829 * tf.to_float(tf.shape(x)[1]) - tf.reduce_sum(tf.sin(tf.abs(x) ** 0.5) * x, axis=1)
    return result


def perceptron_embedding(sizes, sess, coord):
    # init network (first 2 layers)
    input = tf.range(0, sizes[0])

    embed_params = tf.get_variable("perc_embed", shape=[sizes[0], sizes[1]],
                         initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    # initializer=tf.contrib.layers.xavier_initializer())
    top_layer = tf.nn.embedding_lookup(params=embed_params, ids=input)
    # top_layer = tf.get_variable("embed",initializer=tf.truncated_normal(shape=[sizes[0],sizes[1]]))

    # build network
    for i in range(1, len(sizes) - 1):
        name = "perc_fc" + str(i)
        shape = [sizes[i], sizes[i + 1]]
        w = tf.get_variable(name + "_w", shape=shape,
                         initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
        # w = tf.get_variable(name+"_w", shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        # b = tf.get_variable(name+"_b", shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        top_layer = tf.matmul(top_layer, w)
        if i < (len(sizes) - 2):
            top_layer = tf.nn.relu(top_layer)
    # rescale with tg activation
    sess.run(tf.global_variables_initializer())  # FIXME: I need to initialize each of the weights !
    scaling = 1.7159 / tf.maximum(tf.abs(tf.reduce_max(top_layer)), tf.abs(tf.reduce_min(top_layer)))
    scaling_const = sess.run(scaling)
    a_ = [1.0]
    a = tf.get_variable("a", dtype=tf.float32, initializer=tf.constant(a_), trainable=True)
    out = tf.nn.tanh(tf.multiply(scaling_const * top_layer, a))
    b_ = [0.01]
    b = tf.get_variable("b", dtype=tf.float32, initializer=tf.constant(b_), trainable=True)
    out = tf.multiply(out, b)

    """
    max=0
    counter=0


    for i in range(113):
        scaling=1e-10*(1.5**i)
        top=top_layer*scaling
        tdw_ = tf.tanh(top)
        tdw_ = 2 * tdw_
        _tdw = tf.matmul(tf.cosh(top), tf.cosh(top))
        tdw = tf.abs(tf.math.xdivy(tdw_, _tdw))
        mean_=tf.reduce_mean(tf.reduce_mean(tdw,axis=0),axis=0)
        mean=sess.run(mean_)
        if mean > max :
            counter = i
            max = mean
    scaling_const = 1e-10*(1.5**counter)
    out=tf.nn.tanh(top_layer*scaling_const)
    """




    return out


def net3_embed(X, fun_shape, em_shape, sess, coord, tl):
    l0 = tf.expand_dims(X, 1)

    if tl[2] == 1:
        y = True
    else:
        y = False

   # W1 = tf.get_variable("W1", shape=[ fun_shape[0], fun_shape[1]],
    #                     initializer=tf.contrib.layers.xavier_initializer(), trainable=y)
    W1 = tf.get_variable("W1", shape=[fun_shape[0], fun_shape[1]],
                         initializer=tf.contrib.layers.xavier_initializer(), trainable=y)
    l1 = tf.reduce_sum(tf.expand_dims(l0, 3) * W1, axis=2)

    l1_act = tf.nn.relu(l1)

    W2 = eval("perceptron_embedding")(em_shape, sess=sess, coord=coord)
    W2 = tf.reshape(W2, [1, em_shape[0], fun_shape[1], fun_shape[2]])

    l2 = tf.reduce_sum(tf.expand_dims(l1_act, 3) * W2, axis=2)
    l2_act = tf.nn.relu(l2)
    shape = [fun_shape[2], fun_shape[3]]

    W3 = tf.get_variable("W3", shape=[fun_shape[2], fun_shape[3]],
                         initializer=tf.initializers.zeros(), trainable=y)
    l3 = tf.reduce_sum(tf.expand_dims(l2_act, 3) * W3, axis=2)
    return X, l3, [W1, W2, W3]


def test_accuracy(X, x_test,y_test,em_shape ,sess, coord):


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

    yhat_predicted=tf.nn.softmax(yhat_train)


    correct_prediction = tf.equal(tf.argmax(yhat_predicted, 2), tf.argmax(y__, 2))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def _try_params(n_iterations, batch_size, fun_shape, em_shape, db_path, lr, optimizer, scheduler, net3, tl,counter):
    "try some parameters, report testing accuracy with square loss"
    # read data
    x_train, y_train, x_test, y_test = read_dataset(db_path, batch_size)
    # initialize training/testing graph
    # initialize session
    sess = tf.Session()
    coord = tf.train.Coordinator()
    _, yhat_train, X = eval(net3)(X=x_train, fun_shape=fun_shape, em_shape=em_shape, sess=sess, coord=coord, tl=tl)

    y_ = tf.expand_dims(y_train, 1)
    y__=y_

    for i in range(em_shape[0]-1):

        y__=tf.concat([y__,y_],axis=1)

    train_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=yhat_train, labels=y__)

    yhat_predicted=tf.nn.softmax(yhat_train)


    correct_prediction = tf.equal(tf.argmax(yhat_predicted, 2), tf.argmax(y__, 2))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    test_acc_ = test_accuracy(X=X, x_test=x_test, y_test=y_test, em_shape=em_shape, sess=sess, coord=coord)


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
        """
        em_shape=cfg.em_shape
        em_shape[0] = em_shape[0]*(2**i)
        i+=1
        """
        lr=1.56e-7
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
                                                     scheduler=cfg.scheduler, tl=cfg.training_layers, net3="net3_embed",counter=counter)







