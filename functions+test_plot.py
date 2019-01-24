import tensorflow as tf
import time
import inputs
from math import pi
from math import exp
import os,sys,time
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import utils,embedders
from scipy.signal import savgol_filter


def _rescale(x, a, b, c, d):
    """
    Rescales variable from [a, b] to [c, d]
    """
    return c + ((d - c) / (b - a)) * (x - a)


def _MICHALEWICZ(x, xmin=-1, xmax=1):
    """
    https://www.sfu.ca/~ssurjano/michal.html
    """
    data_tensor = _rescale(x, xmin, xmax, 0, pi)
    columns = tf.unstack(data_tensor)


    # Track both the loop index and summation in a tuple in the form (index, summation)

    index = tf.constant(1)
    summation = tf.zeros(tf.shape(x))

    # The loop condition, note the loop condition is 'i < n-1'
    def condition(index, summation):
        return tf.less(index, tf.subtract(tf.shape(columns)[0], 1))

        # The loop body, this will return a result tuple in the same form (index, summation)

    def body(index, summation):
        x_i = tf.gather(columns, index)
        mult_1 = tf.sin(x_i)
        i = tf.to_float(index)
        mult2 = tf.sin((i * x_i ** 2) / pi)
        mult_2 = mult2 ** 20

        summand = tf.add(-1 * mult_1, mult_2)

        return tf.add(index, 1), tf.add(summation, summand)

        # We do not care about the index value here, return only the summation

    result = tf.while_loop(cond=condition, body=body, loop_vars=[index, summation])[1]
    result = tf.reduce_mean(result,axis=1)

    return result


def _rastrigin(x, xmin=-1, xmax=1):
        """
        https://www.sfu.ca/~ssurjano/rastr.html
        """
        x = _rescale(x, xmin, xmax, -5.12, 5.12)

        result = 10.0 * tf.to_float(tf.shape(x)[0])+tf.reduce_sum(x**2 - 10*(tf.cos(2*pi*x)),axis=1)
        return result

def _ROSENBROCK(x, xmin=-1, xmax=1):
        """
        https://www.sfu.ca/~ssurjano/rosen.html
        """
        data_tensor = _rescale(x, xmin, xmax, -2.048, 2.048)
        columns = tf.unstack(data_tensor)

        # Track both the loop index and summation in a tuple in the form (index, summation)
        index=tf.constant(1)
        summation = tf.zeros(tf.shape(x))

        # The loop condition, note the loop condition is 'i < n-1'
        def condition(index, summation):
            return tf.less(index, tf.subtract(tf.shape(columns)[0], 1))

            # The loop body, this will return a result tuple in the same form (index, summation)

        def body(index, summation):
            x_i = tf.gather(columns, index)
            x_ip1 = tf.gather(columns, tf.add(index, 1))

            first_term = (x_ip1 - x_i**2)**2
            second_term = (x_i-1)**2
            summand = tf.add(tf.multiply(100.0, first_term), second_term)

            return tf.add(index, 1), tf.add(summation, summand)

            # We do not care about the index value here, return only the summation

        result=tf.reduce_mean(tf.while_loop(condition, body, [index,summation])[1],axis=1)
        return result

def _stiblinsky_tang(x, xmin=-1, xmax=1):
        """
        https://www.sfu.ca/~ssurjano/stybtang.html
        """
        x = _rescale(x, xmin, xmax, -5, 5)

        result = 0.5*tf.reduce_sum(x**4 - 16*x**2 + 5*x,axis=1)
        return result



def _sphere(x, xmin=-1, xmax=1):
        """
        https://www.sfu.ca/~ssurjano/spheref.html
        """
        x = _rescale(x, xmin, xmax, -5, 5)

        result = tf.reduce_sum(x**2,axis=1)
        return result



def _ackley(x, xmin=-1, xmax=1):
        """
        https://www.sfu.ca/~ssurjano/ackley.html
        """
        a = 20
        b = 0.2
        c = 2 * pi


        d = tf.to_float(tf.shape(x)[0])
        d = 1/d
        print (d)

        x = _rescale(x, xmin, xmax, -32.768, 32.768)

        term1 = -1 * a * tf.math.exp(-b*(d*tf.reduce_sum(x**2,axis=1))**0.5)
        term2 = -1*tf.math.exp(d*tf.reduce_sum(tf.cos(c*x),axis=1))


        return term1 + term2 + a + exp(1)


def _schwefel(x, xmin=-1, xmax=1):
        """
        https://www.sfu.ca/~ssurjano/schwef.html
        """
        x = _rescale(x, xmin, xmax, -500, 500)
        result = 418.9829 * tf.to_float(tf.shape(x)[0]) - tf.reduce_sum(tf.sin(tf.abs(x) ** 0.5) * x,axis=1)
        return result

def _levy(x, xmin=-1, xmax=1):
    """
    https://www.sfu.ca/~ssurjano/levy.html
    """
    x = _rescale(x, xmin, xmax, -10, 10)

    w = 1 + (x - 1)/4
    w1 = w[:, 0]
    ww = w[:, :-1]
    wd = w[:, -1]

    term1 = tf.sin(pi*w1)**2
    termw = tf.reduce_sum((ww**2)*(1+10*tf.sin(pi*ww+1)**2))
    termd = ((wd-1)**2)*(1+tf.sin(2*pi*wd)**2)

    return term1 + termw + termd


out_path = "/home/viacheslav/Documents/9520_final/Plots"
fig = plt.figure()
fig.set_size_inches(12.8, 12.8)
ax = fig.gca(projection='3d')
_x = np.linspace(-1, 1, 100,dtype=np.float32)
_y = np.linspace(-1, 1, 100,dtype=np.float32)
_xv, _yv = np.meshgrid(_x, _y)

# zv = the_function(torch.from_numpy(np.stack([xv, yv], 1))).numpy()

zv = _MICHALEWICZ(np.stack([_xv, _yv], 1))
print (tf.shape(zv))
sess = tf.Session()
_zv = sess.run(zv)
print (_zv)
#print _zv.shape, _xv.shape


surf = ax.plot_surface(_xv, _yv, _zv, rstride=1, cstride=1, cmap=cm.coolwarm, color='c', alpha=0.3, linewidth=0)
#xys = network().detach().numpy()
#zs = the_function(network()).detach().numpy()

plt.show()

plt.savefig(os.path.join(out_path,  "MICHALEWICZ.png"))
plt.close()

