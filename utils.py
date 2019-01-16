import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import collections,os,shutil


def init_unitialized_vars(sess):
    pass


class TrainingMonitor:
    def __init__(self, session_name, out_path, n_ave=30):
        np.set_printoptions(precision=3)

        self.session_name = session_name
        #self.session_id = session_id
        self.n_ave = n_ave
        self.fig_path = os.path.join(out_path, session_name + "_tm.png")
        self.csv_path = os.path.join(out_path, session_name + "_tm.csv")
        self.add_lock = False
        self._records = collections.OrderedDict()
        self._averages = collections.OrderedDict()
        self._b_nums = collections.OrderedDict()
        # remove old plot and text output
        if os.path.exists(self.fig_path):
            shutil.move(self.fig_path, self.fig_path + ".old")
        if os.path.exists(self.csv_path):
            shutil.move(self.csv_path, self.csv_path + ".old")

    def add(self, key, value, b_num):
        if not key in self._records:
            assert not self.add_lock, "can not add any more new variables, the record has been saved"
            self._records[key] = []
            self._averages[key] = []
            self._b_nums[key] = []
        self._records[key].append(value)
        self._averages[key].append(np.average(self._records[key][-self.n_ave:]))
        self._b_nums[key].append(b_num)
        return str(key) + ": " + str(value) + "\tave:" + str(self._averages[key][-1]) + "\t"

    def add_many(self, prefix, val_dict, b_num):
        for key, value in val_dict.iteritems():
            self.add(key=(prefix + key), value=value, b_num=b_num)

    def check_save(self, figure=False, csv=True):
        """
        Saves the model
        :param figure:
        :param csv:
        :param url: send the result to the models server
        :return:
        """
        # block further addition of new variables
        self.add_lock = True
        # compute average/min/max values for all keys
        print "b_num:",  self._b_nums.values()[0][-1],
        for key in self._records:
            print key, #self._records[key][-1],
            print "ave:", "%.3f" % self._averages[key][-1],
        # make a figure
        print
        if figure:
            n_types = len(self._records)
            fig, (ax) = plt.subplots(n_types, 1, figsize=(8, 2 * n_types), dpi=300)
            for i in range(n_types):
                key = self._records.items()[i][0]
                ax[i].plot(self._b_nums[key], self._records[key])
                ax[i].plot(self._b_nums[key], self._averages[key], "ro")
                ax[i].set_ylabel(str(key))
            ax[-1].set_xlabel('Batch Number')
            plt.savefig(self.fig_path)
            plt.close()
        if csv:
            # initialize a csv file if not present
            if not os.path.exists(self.csv_path):
                writer = open(self.csv_path, 'w')
                header = ",".join([str(key[0]) for key in self._records.items()]) + "\n"
                writer.write(header)
                writer.close()
            averages = ",".join([str(averages[-1]) for averages in self._averages.values()]) + "\n"
            writer = open(self.csv_path, 'a')
            writer.write(averages)
            writer.flush()
            writer.close()