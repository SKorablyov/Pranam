import time,os,sys,socket
import tensorflow as tf
import numpy as np
import config_alternative as ca
from keras.datasets import mnist # subroutines for fetching the MNIST dataset
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values


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
    op=str(out_path)
    op=op[-5:]

    if op=="mnist" :
        num_train = 60000  # there are 60000 training examples in MNIST
        num_test = 10000  # there are 10000 test examples in MNIST

        height, width, depth = 28, 28, 1  # MNIST images are 28x28 and greyscale
        num_classes = 10  # there are 10 classes (1 per digit)

        (X_train, y_train), (X_test, y_test) = mnist.load_data()  # fetch MNIST data

        X_train = X_train.reshape(num_train, height * width)  # Flatten data to 1D
        X_test = X_test.reshape(num_test, height * width)  # Flatten data to 1D
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255  # Normalise data to [0, 1] range
        X_test /= 255 # Normalise data to [0, 1] range
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




def read_dataset(db_path,batch_size):
    # load dataset
    machine = socket.gethostname()
    if machine == "viacheslav-HP-Pavilion-Notebook":

        path = str(db_path)
        path=path[2:]
        path = path[:-1]
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

def perceptron_embedding1(sizes,sess,coord):

    # init network (first 2 layers)
    # fixme changed the original implementation
    """
    em_shapes == sizes 256 x 256
    """

    Ws = []
    top_layer = tf.get_variable("perc_embed_pc",
                                shape=[sizes[0],sizes[1]],
                                initializer=tf.contrib.layers.xavier_initializer())
    Ws.append(top_layer)

    """
    Ws==[256x256]
    """
    print(Ws)
    print(top_layer)
    for i in range(1, len(sizes)-1):
        name = "perceptron_fc_pc" + str(i)
        shape = [sizes[i],sizes[i+1]]
        w = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        top_layer = tf.nn.relu(tf.matmul(top_layer,w))
        print(top_layer)
        Ws.append(w)
    return top_layer,Ws

def perceptron_embedding12(sizes,sess,coord):
    Ws=[]
    top_layer = tf.get_variable("perc_embed_pc12",
                                shape=[sizes[0], sizes[1]],
                                initializer=tf.contrib.layers.xavier_initializer())
    Ws.append(top_layer)

    top_layer=tf.nn.relu(top_layer)


    return top_layer,Ws


def perceptron_embedding3(sizes,sess,coord):
    return


def perceptron_embedding2(sizes,sess,coord):
    Ws = []
    i=0
    shape = [sizes[i], sizes[i + 1]]
    name = "perceptron_fc_pc" + str(i)
    w = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    Ws.append(w)
    top_layer = tf.get_variable("perc_embed_pc2",
                                shape=[sizes[0], sizes[1]],
                                initializer=tf.contrib.layers.xavier_initializer())
    top_layer = tf.nn.relu(tf.matmul(top_layer, w))
    Ws.append(top_layer)

    return top_layer,Ws





def actcentron_embedding1(sizes,sess,coord,output_range=3):
    # init network (first 2 layers)
    # fixme changed the original implementation
    Ws = []
    top_layer = tf.get_variable("perc_embed_a1",
                                shape=[sizes[0],sizes[1]],
                                initializer=tf.contrib.layers.xavier_initializer())
    Ws.append(top_layer)
    for i in range(1, len(sizes) - 1):
        name = "perceptron_fc_a" + str(i)
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



def actcentron_embedding12(sizes,sess,coord,output_range=3):
    # init network (first 2 layers)
    # fixme changed the original implementation
    Ws = []
    top_layer = tf.get_variable("perc_embed_a12",
                                shape=[sizes[0],sizes[1]],
                                initializer=tf.contrib.layers.xavier_initializer())
    Ws.append(top_layer)
    sess.run(tf.global_variables_initializer()) # FIXME: I need to initialize each of the weights separately
    tf.train.start_queue_runners(sess,coord)
    scaling = 1.7159 / tf.maximum(tf.abs(tf.reduce_max(top_layer)), tf.abs(tf.reduce_min(top_layer)))
    scaling_const = sess.run(scaling)
    act = output_range * tf.nn.tanh(scaling_const * top_layer)
    return act,Ws


def actcentron_embedding2(sizes,sess,coord,output_range=3):
    # init network (first 2 layers)
    # fixme changed the original implementation
    Ws = []
    i=0
    name = "perceptron_fc_a" + str(i)
    shape = [sizes[i], sizes[i + 1]]
    w = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    Ws.append(w)
    top_layer = tf.get_variable("perc_embed_a2",
                                shape=[sizes[0],sizes[1]],
                                initializer=tf.contrib.layers.xavier_initializer())
    Ws.append(top_layer)
    sess.run(tf.global_variables_initializer()) # FIXME: I need to initialize each of the weights separately
    tf.train.start_queue_runners(sess,coord)
    scaling = 1.7159 / tf.maximum(tf.abs(tf.reduce_max(top_layer)), tf.abs(tf.reduce_min(top_layer)))
    scaling_const = sess.run(scaling)
    act = output_range * tf.nn.tanh(scaling_const * top_layer)
    return act,Ws




def net3_embed_1(X,fun_shape,em,em_shape,sess,coord,tl):

    l0 = tf.expand_dims(X, 1)

    if tl[2]==1: y=True
    else: y=0

    em_str=str(em)

    if tl[0]==1 :
        em_1=str(em_str)+"1"
        W1, PWs = eval(em_1)(em_shape, sess=sess, coord=coord)
        W1 = tf.reshape(W1, [1, em_shape[0], fun_shape[0], fun_shape[1]])
        l1 = tf.reduce_sum(tf.expand_dims(l0, 3) * W1, axis=2)
        l1_act = tf.nn.relu(l1)
        if tl[1]==1:
            em_1=str(em_str)+"12"
            em_shape_1 = em_shape
            em_shape_1[1] = fun_shape[1] * fun_shape[2]
            W2, _ = eval(em_1)(em_shape_1, sess=sess, coord=coord)
            W2 = tf.reshape(W2, [1, em_shape_1[0], fun_shape[1], fun_shape[2]])
            l2 = tf.reduce_sum(tf.expand_dims(l1_act, 3) * W2, axis=2)
            l2_act = tf.nn.relu(l2)
        else:
            W2 = tf.get_variable("W2", shape=[em_shape[0], fun_shape[0], fun_shape[1]],
                                 initializer=tf.contrib.layers.xavier_initializer(), trainable=False)
            l2 = tf.reduce_sum(tf.expand_dims(l1_act, 3) * W2, axis=2)
            l2_act = tf.nn.relu(l2)

    else:
        em_1 = str(em_str) + "2"

        W1 = tf.get_variable("W1", shape=[em_shape[0], fun_shape[0], fun_shape[1]],
                             initializer=tf.contrib.layers.xavier_initializer(), trainable=False)
        l1 = tf.reduce_sum(tf.expand_dims(l0, 3) * W1, axis=2)
        l1_act = tf.nn.relu(l1)
        em_shape_1 = em_shape
        em_shape_1[1] = fun_shape[1] * fun_shape[2]

        W2, PWs = eval(em_1)(em_shape_1, sess=sess, coord=coord)
        W2 = tf.reshape(W2, [1, em_shape_1[0], fun_shape[1], fun_shape[2]])
        l2 = tf.reduce_sum(tf.expand_dims(l1_act, 3) * W2, axis=2)
        l2_act = tf.nn.relu(l2)


def net3_embed(X, fun_shape, em, em_shape, sess, coord, tl,em_dif):

        l0 = tf.expand_dims(X, 1)

        if tl[2] == 1:
            y = True
        else:
            y = 0

        em_str = str(em)
        em_str = em_str[2:]
        em_str = em_str[:-1]

        if tl[0] == 1:
            print(em)
            print(em_dif)
            if em_dif==1:em_str="perceptron_embedding"
            elif em_dif==2:em_str="actcentron_embedding"
            em_1 = str(em_str) + "1"
            W1, PWs = eval(em_1)(em_shape, sess=sess, coord=coord)
            W1 = tf.reshape(W1, [1, em_shape[2], fun_shape[0], fun_shape[1]])
            l1 = tf.reduce_sum(tf.expand_dims(l0, 3) * W1, axis=2)
            l1_act = tf.nn.relu(l1)
            W2 = tf.get_variable("W2", shape=[em_shape[0], fun_shape[0], fun_shape[1]],
                                 initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
            l2 = tf.reduce_sum(tf.expand_dims(l1_act, 3) * W2, axis=2)
            l2_act = tf.nn.relu(l2)


        else:
            em_1 = str(em_str) + "2"

            W1 = tf.get_variable("W1", shape=[em_shape[0], fun_shape[0], fun_shape[1]],
                                 initializer=tf.contrib.layers.xavier_initializer(), trainable=False)
            l1 = tf.reduce_sum(tf.expand_dims(l0, 3) * W1, axis=2)
            l1_act = tf.nn.relu(l1)
            em_shape_1 = em_shape
            em_shape_1[1] = fun_shape[1] * fun_shape[2]

            W2, PWs = eval(em_1)(em_shape_1, sess=sess, coord=coord)
            W2 = tf.reshape(W2, [1, em_shape_1[0], fun_shape[1], fun_shape[2]])
            l2 = tf.reduce_sum(tf.expand_dims(l1_act, 3) * W2, axis=2)
            l2_act = tf.nn.relu(l2)







        W3 = tf.get_variable("W3", shape=[fun_shape[2], fun_shape[3]],
                         initializer=tf.contrib.layers.xavier_initializer(),trainable=y)
        l3 = tf.reduce_sum(tf.expand_dims(l2_act, 3) * W3, axis=2)
        return X, l3, PWs + [W1,W2,W3]
"""

def net3_embed(X, fun_shape, em, em_shape, sess, coord, tl,em_dif):
    l0 = tf.expand_dims(X, 1)
    W1 = tf.get_variable("W1", shape=[em_shape[0], fun_shape[0], fun_shape[1]],
                         initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    PWs = []
    PWs.append(W1)
    l1 = tf.reduce_sum(tf.expand_dims(l0, 3) * W1, axis=2)
    l1_act = tf.nn.relu(l1)

    em_shape_1 = em_shape
    em_shape_1[1] = fun_shape[1] * fun_shape[2]
    W2 = tf.get_variable("W2", shape=[em_shape[0], fun_shape[0], fun_shape[1]],
                         initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    l2 = tf.reduce_sum(tf.expand_dims(l1_act, 3) * W2, axis=2)
    l2_act = tf.nn.relu(l2)

    W3 = tf.get_variable("W3", shape=[fun_shape[2], fun_shape[3]],
                         initializer=tf.contrib.layers.xavier_initializer(), trainable=False)
    l3 = tf.reduce_sum(tf.expand_dims(l2_act, 3) * W3, axis=2)
    return X, l3, PWs + [W1, W2, W3]
"""
def _try_params(n_iterations,batch_size,fun_shape,em,em_shape,db_path,lr,optimizer,scheduler,net3,tl,em_dif):
    "try some parameters, report testing accuracy with square loss"
    # read data
    x_train, y_train, x_test, y_test = read_dataset(db_path, batch_size)
    # initialize training/testing graph
    # initialize session
    sess = tf.Session()
    coord = tf.train.Coordinator()
    _,yhat_train,_ = eval(net3)(X=x_train,fun_shape=fun_shape,em=em,em_shape=em_shape,sess=sess,coord=coord,tl=tl,em_dif=em_dif)
    if batch_size!=1:
        y_diff = tf.expand_dims(y_train, 1) - yhat_train

        train_loss = tf.reduce_mean(tf.reduce_mean(y_diff ** 2, axis=2), axis=0)
    else:
        shape=tf.shape(y_train)
        yhat_train_1=tf.reshape(yhat_train,shape)
        y_train_ln=tf.log(yhat_train_1)
        train_loss_tensor=tf.subtract(y_train_ln,y_train)
        train_loss=tf.reduce_sum(train_loss_tensor)
        train_loss=tf.negative(train_loss)
        print(train_loss)

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
        if i %500==1:
            # print "argmin of the train loss", _y_diff[_train_loss.argmin()],
            print ("step:",i,"mean_loss", np.mean(_train_loss), "min_loss", np.min(_train_loss),)
            # print "y_train:", np.mean(_y_train), np.var(_y_train),_y_train.shape,
            # print "y_hat:", np.mean(_yhat_train),np.var(_yhat_train), _yhat_train.shape,
            print ("lr",lr)
            print("exps:" , cfg.batch_size   / (time.time() - start))
        # history
        if i % 1000 == 1:
            _train_loss, = sess.run([train_loss],feed_dict={lr_current:lr})
    sess.close()
    tf.reset_default_graph()
    sel_point = np.mean(_train_losses[-200:-100], axis=0).argmin()
    minmean_loss = np.mean(_train_losses[:-100], axis=0)[sel_point]
    loss_hist = np.asarray(_train_losses)[:,sel_point]
    return minmean_loss,loss_hist


def try_params(n_iterations,batch_size,fun_shape,em,em_shape,db_path,lr,optimizer,scheduler,net3,tl,em_dif):
    train_cost, train_cost_hist = tf.py_func(_try_params,
                                            [n_iterations,batch_size,fun_shape,em,em_shape,db_path,lr,optimizer,scheduler,net3,tl,em_dif],
                                            [tf.float32,tf.float32])
    sess = tf.Session()
    _train_cost,_train_cost_hist = sess.run([train_cost,train_cost_hist])
    sess.close()
    return _train_cost,_train_cost_hist


if __name__ == "__main__":

          # set up the config and folders
          machine = socket.gethostname()
          cfg = ca.cfg4_39
          config_name=cfg.name
          counter=-1
          tl=cfg.training_layers
          #em_dif_list_2=["cfg4_44","cfg4_45","cfg_mnist_a32"]
          em_dif_list_2 = ["cfg4_46", "cfg4_47","cfg_mnist_gd32"]
          em_dif_list_1=[]
          em_dif_list_2=[]
          if config_name in em_dif_list_1 :em_dif=1
          elif config_name in em_dif_list_2 : em_dif=2
          else: em_dif=0

          if not os.path.exists(cfg.out_path): os.makedirs(cfg.out_path)
          if machine!="viacheslav-HP-Pavilion-Notebook":
              for lr in cfg.lrs:
                  counter += 1
                  train_costs = []
                  train_cost_hists = []
                  net3 = "net3_embed_1"
                  for i in range(cfg.n_runs):
                      # generate dataset
                      #generate_dataset(cfg.db_path, cfg.genf_shape, cfg.train_samples, cfg.test_samples,noise=cfg.noise)

                      train_cost, train_cost_hist = try_params(n_iterations=cfg.n_iterations,
                                                               batch_size=cfg.batch_size,
                                                               fun_shape=cfg.fun_shape,
                                                               em=cfg.em,
                                                               em_shape=cfg.em_shape,
                                                               db_path=cfg.db_path, lr=lr, optimizer=cfg.optimizer,
                                                               scheduler=cfg.scheduler, net3=net3, tl=tl,em_dif=em_dif)
                      print("\n", "train cost", train_cost)
                      train_costs.append(train_cost)
                      train_cost_hists.append(train_cost_hist)
                      train_costs = np.asarray(train_costs)
                      train_cost_hists = np.asarray(train_cost_hists)
                      train_cost_hist = np.mean(train_cost_hists, axis=0)
                      np.savetxt(
                          os.path.join(cfg.out_path, config_name + "_train_cost_hist_lr_" + str(lr) + str(counter)),
                          train_cost_hist)
          else:
              for lr in cfg.lrs:
                  counter += 1
                  train_costs = []
                  train_cost_hists = []
                  net3 = "net3_embed"
                  for i in range(cfg.n_runs):
                      # generate dataset
                      #generate_dataset(cfg.db_path, cfg.genf_shape, cfg.train_samples, cfg.test_samples,noise=cfg.noise)

                      train_cost, train_cost_hist = try_params(n_iterations=cfg.n_iterations,
                                                               batch_size=cfg.batch_size,
                                                               fun_shape=cfg.fun_shape,
                                                               em=cfg.em,
                                                               em_shape=cfg.em_shape,
                                                               db_path=cfg.db_path, lr=lr, optimizer=cfg.optimizer,
                                                               scheduler=cfg.scheduler, net3=net3, tl=tl,em_dif=em_dif)
                      print("\n", "train cost", train_cost)
                      train_costs.append(train_cost)
                      train_cost_hists.append(train_cost_hist)
                      train_costs = np.asarray(train_costs)
                      train_cost_hists = np.asarray(train_cost_hists)
                      train_cost_hist = np.mean(train_cost_hists, axis=0)
                      np.savetxt(
                          os.path.join(cfg.out_path, config_name + "_train_cost_hist_lr_" + str(lr) + str(counter)),
                          train_cost_hist)







