import numpy as np
import socket, time, os


class cfg:
    # dataset parameters (default)
    db_path = "/home/maksym/Desktop/SLT/data"
    out_path = "/home/maksym/Desktop/SLT/plots"
    f_shape = [16, 16, 3]
    train_samples = 10000
    test_samples = 100000
    train_steps = 20000
    lrs = [1e-3]
    noise = None
    n_runs = 10

    # hyperband parameters
    batch_size = 100

    # square exhaustive plot parameters
    n_repeats = 20
    l2_vals = 2 ** np.arange(7)
    l3_vals = 2 ** np.arange(7)




class cfg_a1:
        name = "cfg4_a1"
        machine = socket.gethostname()
        if machine == "Ikarus":
            db_path = "/home/maksym/Desktop/slt/cfg4_1"
            out_path = "/home/maksym/Desktop/slt/"
        elif machine.startswith("matlaber"):
            db_path = "/mas/u/mkkr/mk/slt/cfg4_1"
            out_path = "/mas/u/mkkr/mk/slt/"
        elif machine == "viacheslav-HP-Pavilion-Notebook":
            db_path = "/home/viacheslav/Documents/9520_final/Datasets/cfg4_32_fat"
            out_path = "/home/viacheslav/Documents/9520_final/Datasets/"
        else:
            raise Exception("path not set up on machine")
        db_path = os.path.join(db_path)
        out_path = os.path.join(out_path, name)

        training_layers=[0,1,0]
        genf_shape = [64, 64, 3]
        noise = None
        train_samples = 1000000
        test_samples = 1000000
        lrs = [1e-3]
        n_runs = 10
        n_iterations = 10000
        batch_size = 100
        fun_shape = [64, 64, 64, 12]
        em = "actcentron_embedding"
        em_shape = [256, 1024]

        # scheduler
        scheduler = "none"
        # optimizer
        optimizer = "tf.train.AdamOptimizer"