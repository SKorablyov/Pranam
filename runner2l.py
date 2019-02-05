import tensorflow as tf
import numpy as np
import sys, os
import utils, networks  # ,config
from config import *
import time
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from pranam import PranamOptimizer
from examples import run_pranam

from examples import run_pranam2
from examples import run_mod_pranam
from temp3 import calc

if __name__ == '__main__':
    lrs=[0.1,0.01,0.001]

    shape = [1,1,1]
    shapes = []
    shapes.append(shape)


    for lr in lrs :
        counter = 0
        for shape in shapes:
            res =[]
            for t in range(2):
                print t
                sess = tf.Session()
                result ,_= calc(shape=shape,lr=lr,sess = sess)
                sess.close()
                tf.reset_default_graph()
                res.append(result)
            np.asarray(res)
            result1 = np.mean(res, axis=0)
            file_name = 'mean_loss_' + str(counter) + '_' + str(lr) + '_.txt'
            my_dir = "./results"
            fname = os.path.join(my_dir, file_name)
            with open(fname, 'w') as f:
                for item in result1:
                    f.write("%s\n" % item)









