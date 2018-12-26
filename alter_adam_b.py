import time,os,sys,socket
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import config_alternative as ca

def hierarchical_compositional(genf_shape,n_samples,W=None,noise=None):
    "generate a bunch of samples, the activation function is frozen as relu"
    X = np.matrix(np.asarray(np.random.uniform(size=[n_samples,genf_shape[0]])))
    lastX = X
    # initialize
    if W is None:
        W = []
        for i in np.arange(len(genf_shape) - 1) + 1:
            w = np.matrix(np.random.uniform(low=-2, high=2, size=[genf_shape[i - 1], genf_shape[i]]))
            W.append(w)
    # apply weights, get Ys
    for i in np.arange(len(genf_shape) - 1):
        print ("layer", i, "incoming shape", lastX.shape)
        # print "test shape", lastX.shape,W[i].shape, (lastX * W[i]).shape, np.maximum(lastX * W[i],0).shape
        lastX = np.maximum(np.asarray(lastX * W[i]),0)
    Y = lastX
    if noise is not None:
        Y = Y + np.random.uniform(size=Y.shape,low=-0.5*noise,high=0.5*noise)
    return X,Y,W


def generate_dataset(out_path, f_shape, train_samples, test_samples, noise):
    if not os.path.exists(out_path): os.makedirs(out_path)
    # generate training set
    X_train,Y_train,W = hierarchical_compositional(f_shape, n_samples=train_samples, noise=noise)
    np.save(os.path.join(out_path, "X_train"), np.asarray(X_train,np.float32))
    np.save(os.path.join(out_path, "Y_train"), np.asarray(Y_train,np.float32))
    # generate testing set
    X_test,Y_test,_ = hierarchical_compositional(f_shape,n_samples=test_samples, W=W, noise=noise)
    np.save(os.path.join(out_path, "X_test"), np.asarray(X_test,np.float32))
    np.save(os.path.join(out_path, "Y_test"), np.asarray(Y_test,np.float32))
    print ("data has been saved")


def read_dataset(db_path,batch_size):
    # load dataset
    machine = socket.gethostname()
    if machine == "viacheslav-HP-Pavilion-Notebook":
        path = "/home/viacheslav/Documents/9520_final/Datasets/cfg4_a"
        X_train = np.load(os.path.join(path, "X_train.npy"))
        Y_train = np.load(os.path.join(path, "Y_train.npy"))
        X_test = np.load(os.path.join(path, "X_test.npy"))
        Y_test = np.load(os.path.join(path, "Y_test.npy"))

    else :
        X_train = np.load(os.path.join(db_path, "X_train.npy"))
        Y_train = np.load(os.path.join(db_path, "Y_train.npy"))
        X_test = np.load(os.path.join(db_path, "X_test.npy"))
        Y_test = np.load(os.path.join(db_path, "Y_test.npy"))

    print ("loaded dataset of shapes:", X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
    # make batches
    x_train, y_train = tf.train.slice_input_producer([X_train,Y_train])
    x_train, y_train = tf.train.batch([x_train,y_train],batch_size)
    x_test, y_test = tf.train.slice_input_producer([X_test, Y_test])
    x_test, y_test = tf.train.batch([x_test, y_test], batch_size)
    return x_train,y_train,x_test,y_test


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
    #result = x.abs().sqrt().sin().mul(x).sum(1).mul(-1).add(418.9829 * x.size(1))
    result = 418.9829 * tf.to_float(tf.shape(x)[1]) - tf.reduce_sum(tf.sin(tf.abs(x)**0.5) * x,axis=1)
    return result

def perceptron_embedding(sizes,sess,coord):
    # init network (first 2 layers)
    # fixme changed the original implementation
    Ws = []
    top_layer = tf.get_variable("perc_embed",
                                shape=[sizes[0],sizes[1]],
                                initializer=tf.contrib.layers.xavier_initializer())
    Ws.append(top_layer)
    for i in range(1, len(sizes) - 1):
        name = "perceptron_fc" + str(i)
        shape = [sizes[i],sizes[i+1]]
        w = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        top_layer = tf.nn.relu(tf.matmul(top_layer,w))
        Ws.append(w)
    return top_layer,Ws

def actcentron_embedding(sizes,sess,coord,output_range=3):
    # init network (first 2 layers)
    # fixme changed the original implementation
    Ws = []
    top_layer = tf.get_variable("perc_embed",
                                shape=[sizes[0],sizes[1]],
                                initializer=tf.contrib.layers.xavier_initializer())
    Ws.append(top_layer)
    for i in range(1, len(sizes) - 1):
        name = "perceptron_fc" + str(i)
        shape = [sizes[i],sizes[i+1]]
        w = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        top_layer = tf.nn.relu(tf.matmul(top_layer,w))
        Ws.append(w)
    sess.run(tf.global_variables_initializer()) # FIXME: I need to initialize each of the weights separately
    tf.train.start_queue_runners(sess,coord)
    scaling = 1.7159 / tf.maximum(tf.abs(tf.reduce_max(top_layer)), tf.abs(tf.reduce_min(top_layer)))
    scaling_const = sess.run(scaling)
    act = output_range * tf.nn.tanh(scaling_const * top_layer)
    return act,Ws

def actcentron_embedding_a(sizes,sess,coord,output_range=3):
    # init network (first 2 layers)
    # fixme changed the original implementation
    Ws = []
    top_layer = tf.get_variable("perc_embed",
                                shape=[sizes[1],sizes[1]],
                                initializer=tf.contrib.layers.xavier_initializer())
    Ws.append(top_layer)
    for i in range(1, len(sizes) - 1):
        name = "perceptron_fc" + str(i)
        shape = [sizes[i],sizes[i+1]]
        w = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        top_layer = tf.nn.relu(tf.matmul(top_layer,w))
        Ws.append(w)
    sess.run(tf.global_variables_initializer()) # FIXME: I need to initialize each of the weights separately
    tf.train.start_queue_runners(sess,coord)
    scaling = 1.7159 / tf.maximum(tf.abs(tf.reduce_max(top_layer)), tf.abs(tf.reduce_min(top_layer)))
    scaling_const = sess.run(scaling)
    act = output_range * tf.nn.tanh(scaling_const * top_layer)
    return act,Ws




def net3_embed1(X,fun_shape,em,em_shape,sess,coord):
    "3-layer network with nonconvex embedding"
    # [16,16,16,3]
    # W [ batch, surface_dots, w_in, w_out]
    l0 = tf.expand_dims(X, 1)
    W1,PWs = eval(em)(em_shape,sess=sess,coord=coord) # [120, 60, 256]
    W1 = tf.reshape(W1,[1,em_shape[0],fun_shape[0],fun_shape[1]]) # 16,16
    l1 = tf.reduce_sum(tf.expand_dims(l0,3) * W1,axis=2)
    l1_act = tf.nn.relu(l1)
    W2 = tf.get_variable("W2", shape=[fun_shape[1], fun_shape[2]],
                         initializer=tf.contrib.layers.xavier_initializer(),trainable=False)
    l2 = tf.reduce_sum(tf.expand_dims(l1_act, 3) * W2, axis=2)
    l2_act = tf.nn.relu(l2)
    W3 = tf.get_variable("W3", shape=[fun_shape[2], fun_shape[3]],
                         initializer=tf.contrib.layers.xavier_initializer(),trainable=False)
    l3 = tf.reduce_sum(tf.expand_dims(l2_act, 3) * W3, axis=2)
    return X, l3, PWs + [W1,W2,W3]

def net3_embed2(X,fun_shape,em,em_shape,sess,coord):
    "3-layer network with nonconvex embedding"
    # [16,16,16,3]
    # W [ batch, surface_dots, w_in, w_out]
    l0 = tf.expand_dims(X, 1)
    #print(W1)
    W1 = tf.get_variable("W1", shape=[em_shape[0], fun_shape[0], fun_shape[1]],
                         initializer=tf.contrib.layers.xavier_initializer(), trainable=False)
    l1 = tf.reduce_sum(tf.expand_dims(l0,3) * W1,axis=2)
    l1_act = tf.nn.relu(l1)
    W2, PWs = eval(em)(em_shape, sess=sess, coord=coord)  # [120, 60, 256]
    W2 = tf.reshape(W2, [fun_shape[0], fun_shape[1]])  # 16,16
    #print(W2)
    l2 = tf.reduce_sum(tf.expand_dims(l1_act, 3) * W2, axis=2)
    l2_act = tf.nn.relu(l2)
    W3 = tf.get_variable("W3", shape=[fun_shape[2], fun_shape[3]],
                         initializer=tf.contrib.layers.xavier_initializer(),trainable=False)
    l3 = tf.reduce_sum(tf.expand_dims(l2_act, 3) * W3, axis=2)
    return X, l3, PWs + [W1,W2,W3]


def net3_embed12(X,fun_shape,em,em_shape,sess,coord):
    "3-layer network with nonconvex embedding"
    # [16,16,16,3]
    # W [ batch, surface_dots, w_in, w_out]
    l0 = tf.expand_dims(X, 1)
    W1,PWs = eval(em)(em_shape,sess=sess,coord=coord) # [120, 60, 256]
    W1 = tf.reshape(W1,[1,em_shape[0],fun_shape[0],fun_shape[1]]) # 16,16
    l1 = tf.reduce_sum(tf.expand_dims(l0,3) * W1,axis=2)
    l1_act = tf.nn.relu(l1)
    W2 = tf.get_variable("W2", shape=[fun_shape[1], fun_shape[2]],
                         initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
    l2 = tf.reduce_sum(tf.expand_dims(l1_act, 3) * W2, axis=2)
    l2_act = tf.nn.relu(l2)
    W3 = tf.get_variable("W3", shape=[fun_shape[2], fun_shape[3]],
                         initializer=tf.contrib.layers.xavier_initializer(),trainable=False)
    l3 = tf.reduce_sum(tf.expand_dims(l2_act, 3) * W3, axis=2)
    return X, l3, PWs + [W1,W2,W3]

def net3_embed12_1(X,fun_shape,em,em_shape,sess,coord):
    "3-layer network with nonconvex embedding"
    # [16,16,16,3]
    # W [ batch, surface_dots, w_in, w_out]
    l0 = tf.expand_dims(X, 1)
    W1,PWs = eval(em)(em_shape,sess=sess,coord=coord) # [120, 60, 256]
    W1 = tf.reshape(W1,[1,em_shape[0],fun_shape[0],fun_shape[1]]) # 16,16
    l1 = tf.reduce_sum(tf.expand_dims(l0,3) * W1,axis=2)
    l1_act = tf.nn.relu(l1)
    W2, PWs = eval(em)(em_shape, sess=sess, coord=coord)  # [120, 60, 256]
    W2 = tf.reshape(W2, [fun_shape[0], fun_shape[1]])  # 16,16
    l2 = tf.reduce_sum(tf.expand_dims(l1_act, 3) * W2, axis=2)
    l2_act = tf.nn.relu(l2)
    W3 = tf.get_variable("W3", shape=[fun_shape[2], fun_shape[3]],
                         initializer=tf.contrib.layers.xavier_initializer(),trainable=False)
    l3 = tf.reduce_sum(tf.expand_dims(l2_act, 3) * W3, axis=2)
    return X, l3, PWs + [W1,W2,W3]



def _try_params(n_iterations,batch_size,fun_shape,em,em_shape,db_path,lr,optimizer,scheduler,net3):
    "try some parameters, report testing accuracy with square loss"
    # read data
    x_train, y_train, x_test, y_test = read_dataset(db_path, batch_size)
    # initialize training/testing graph
    # initialize session
    sess = tf.Session()
    coord = tf.train.Coordinator()
    _,yhat_train,Ws = eval(net3)(X=x_train,fun_shape=fun_shape,em=em,em_shape=em_shape,sess=sess,coord=coord)
    y_diff = tf.expand_dims(y_train,1) - yhat_train

    train_loss = tf.reduce_mean(tf.reduce_mean(y_diff**2,axis=2),axis=0)
    lr_current = tf.placeholder(tf.float32)
    train_step = eval(optimizer)(learning_rate=lr_current).minimize(train_loss)

    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess,coord)
    _train_losses = []
    for i in range(n_iterations):
        start = time.time()
        _train_loss,_ = sess.run([train_loss,train_step],feed_dict={lr_current:lr})
        _train_losses.append(_train_loss)
        if scheduler=="none":
            pass
        elif scheduler == "dlp":
            # scheduler the step size
            if i % 2000 == 1999:
                if np.mean(_train_losses[-1000:]) >= np.mean(_train_losses[-2000:-1000]):
                    lr = lr * 0.5
        else:
            machine = socket.gethostname()
            if machine!="viacheslav-HP-Pavilion-Notebook":
                raise ValueError("unknown scheduler")

            else:
                pass

        # printing
        if i % 100 == 1:
            # print "argmin of the train loss", _y_diff[_train_loss.argmin()],
            print ("step:",i,"mean_loss", np.mean(_train_loss), "min_loss", np.min(_train_loss),)
            # print "y_train:", np.mean(_y_train), np.var(_y_train),_y_train.shape,
            # print "y_hat:", np.mean(_yhat_train),np.var(_yhat_train), _yhat_train.shape,
            print ("lr",lr)
            print("exps:" , cfg.batch_size   / (time.time() - start))
        # history
        if i % 100 == 1:
            _train_loss, = sess.run([train_loss],feed_dict={lr_current:lr})
    sess.close()
    tf.reset_default_graph()
    sel_point = np.mean(_train_losses[-200:-100], axis=0).argmin()
    minmean_loss = np.mean(_train_losses[:-100], axis=0)[sel_point]
    loss_hist = np.asarray(_train_losses)[:,sel_point]
    return minmean_loss,loss_hist


def try_params(n_iterations,batch_size,fun_shape,em,em_shape,db_path,lr,optimizer,scheduler,net3):
    train_cost, train_cost_hist = tf.py_func(_try_params,
                                            [n_iterations,batch_size,fun_shape,em,em_shape,db_path,lr,optimizer,scheduler,net3],
                                            [tf.float32,tf.float32])
    sess = tf.Session()
    _train_cost,_train_cost_hist = sess.run([train_cost,train_cost_hist])
    sess.close()
    return _train_cost,_train_cost_hist


if __name__ == "__main__":

    # set up the config and folders
    model_list=["cfg_a1","cfg_a2","cfg_a12","cfg_b1","cfg_b2","cfg_b3"]
    for cfg_name in model_list :
        cfg = ca.cfg_a3
        config_name = "cfg_a3"
        if ca.cfg_a1.training_layers[0] == 1:
            em = cfg.em

        if not os.path.exists(cfg.out_path): os.makedirs(cfg.out_path)
        if cfg.training_layers[0] == 1:
            if cfg.training_layers[1] == 1:
                for lr in cfg.lrs:
                    train_costs = []
                    train_cost_hists = []
                    net3 = "net3_embed12"
                    for i in range(cfg.n_runs):
                        # generate dataset
                        # generate_dataset(cfg.db_path, cfg.genf_shape, cfg.train_samples, cfg.test_samples, noise=cfg.noise)
                        # train_cost,train_cost_hist = try_params(1000,cfg.batch_size,[64,32,1],cfg.db_path,cfg.test_samples, lr=lr)
                        train_cost, train_cost_hist = try_params(n_iterations=cfg.n_iterations,
                                                                 batch_size=cfg.batch_size,
                                                                 fun_shape=cfg.fun_shape,
                                                                 em=cfg.em,
                                                                 em_shape=cfg.em_shape,
                                                                 db_path=cfg.db_path, lr=lr, optimizer=cfg.optimizer,
                                                                 scheduler=cfg.scheduler, net3=net3)
                        print("\n", "train cost", train_cost)
                        train_costs.append(train_cost)
                        train_cost_hists.append(train_cost_hist)
            else:
                for lr in cfg.lrs:
                    train_costs = []
                    train_cost_hists = []
                    net3 = "net3_embed1"
                    for i in range(cfg.n_runs):
                        # generate dataset
                        # generate_dataset(cfg.db_path, cfg.genf_shape, cfg.train_samples, cfg.test_samples, noise=cfg.noise)
                        # train_cost,train_cost_hist = try_params(1000,cfg.batch_size,[64,32,1],cfg.db_path,cfg.test_samples, lr=lr)
                        train_cost, train_cost_hist = try_params(n_iterations=cfg.n_iterations,
                                                                 batch_size=cfg.batch_size,
                                                                 fun_shape=cfg.fun_shape,
                                                                 em=cfg.em,
                                                                 em_shape=cfg.em_shape,
                                                                 db_path=cfg.db_path, lr=lr, optimizer=cfg.optimizer,
                                                                 scheduler=cfg.scheduler, net3=net3)
                        print("\n", "train cost", train_cost)
                        train_costs.append(train_cost)
                        train_cost_hists.append(train_cost_hist)
        else:
            for lr in cfg.lrs:
                train_costs = []
                train_cost_hists = []
                net3 = "net3_embed2"
                for i in range(cfg.n_runs):
                    # generate dataset
                    # generate_dataset(cfg.db_path, cfg.genf_shape, cfg.train_samples, cfg.test_samples, noise=cfg.noise)
                    # train_cost,train_cost_hist = try_params(1000,cfg.batch_size,[64,32,1],cfg.db_path,cfg.test_samples, lr=lr)
                    train_cost, train_cost_hist = try_params(n_iterations=cfg.n_iterations,
                                                             batch_size=cfg.batch_size,
                                                             fun_shape=cfg.fun_shape,
                                                             em=cfg.em,
                                                             em_shape=cfg.em_shape,
                                                             db_path=cfg.db_path, lr=lr, optimizer=cfg.optimizer,
                                                             scheduler=cfg.scheduler, net3=net3)
                    print("\n", "train cost", train_cost)
                    train_costs.append(train_cost)
                    train_cost_hists.append(train_cost_hist)
                    train_cost_hists.append(train_cost_hist)

        train_costs = np.asarray(train_costs)
        train_cost_hists = np.asarray(train_cost_hists)
        train_cost_hist = np.mean(train_cost_hists, axis=0)
        np.savetxt(os.path.join(cfg.out_path, config_name + "_train_cost_hist_lr_" + str(lr)), train_cost_hist)



