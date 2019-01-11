import numpy as np
import socket, time, os

class cfg_sp_2_2:
        name = "cfg_sp_2_2"
        machine = socket.gethostname()
        if machine == "Ikarus":
            db_path = "/home/maksym/Desktop/slt/cfg_spolynom"
            out_path = "/home/maksym/Desktop/slt/"
        elif machine.startswith("matlaber"):
            db_path = "/mas/u/mkkr/mk/slt/cfg_spolynom"
            out_path = "/mas/u/mkkr/mk/slt/"
        elif machine == "viacheslav-HP-Pavilion-Notebook":
            db_path = "/home/viacheslav/Documents/9520_final/Datasets/cfg_spolynom"
            out_path = "/home/viacheslav/Documents/9520_final/Datasets/"
        else:
            raise Exception("path not set up on machine")
        db_path = os.path.join(db_path)
        out_path = os.path.join(out_path, name)

        training_layers=[0,1,0]
        genf_shape = [16, 16, 1]
        noise = None
        train_samples = 1000000
        test_samples = 1000000
        lrs = [1e-4, 1e-5, 2e-5, 4e-5, 8e-5, 1e-6, 2e-6, 4e-6, 8e-6, ]
        n_runs = 1
        n_iterations = 50000
        batch_size = 100
        fun_shape = [16, 16, 16, 1]
        em_shape = [256, 256]

        optimizer="tf.train.GradientDescentOptimizer"
        scheduler="none"


class cfg_sp_2_3:
    name = "cfg_sp_2_3"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg_spolynom"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg_spolynom"
        out_path = "/mas/u/mkkr/mk/slt/"
    elif machine == "viacheslav-HP-Pavilion-Notebook":
        db_path = "/home/viacheslav/Documents/9520_final/Datasets/cfg_spolynom"
        out_path = "/home/viacheslav/Documents/9520_final/Datasets/"
    else:
        raise Exception("path not set up on machine")
    db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)

    training_layers = [0, 1, 0]
    genf_shape = [16, 16, 1]
    noise = None
    train_samples = 1000000
    test_samples = 1000000
    lrs = [1e-4,  1e-5, 2e-5, 4e-5, 8e-5, 1e-6, 2e-6, 4e-6, 8e-6, ]
    n_runs = 5
    n_iterations = 50000
    batch_size = 100
    fun_shape = [16, 16, 16, 1]
    em_shape = [256, 256,256]

    optimizer = "tf.train.GradientDescentOptimizer"
    scheduler = "none"


class cfg_sp_12_2:
    name = "cfg_sp_12_2"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg_spolynom"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg_spolynom"
        out_path = "/mas/u/mkkr/mk/slt/"
    elif machine == "viacheslav-HP-Pavilion-Notebook":
        db_path = "/home/viacheslav/Documents/9520_final/Datasets/cfg_spolynom"
        out_path = "/home/viacheslav/Documents/9520_final/Datasets/"
    else:
        raise Exception("path not set up on machine")
    db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)

    training_layers = [1, 1, 1]
    genf_shape = [16, 16, 1]
    noise = None
    train_samples = 1000000
    test_samples = 1000000
    lrs = [1e-4, 1e-5, 2e-5, 4e-5, 8e-5, 1e-6, 2e-6, 4e-6, 8e-6, ]
    n_runs = 1
    n_iterations = 50000
    batch_size = 100
    fun_shape = [16, 16, 16, 1]
    em_shape = [256, 256]

    optimizer = "tf.train.GradientDescentOptimizer"
    scheduler = "none"


class cfg_sp_12_3:
    name = "cfg_sp_12_3"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg_spolynom"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg_spolynom"
        out_path = "/mas/u/mkkr/mk/slt/"
    elif machine == "viacheslav-HP-Pavilion-Notebook":
        db_path = "/home/viacheslav/Documents/9520_final/Datasets/cfg_spolynom"
        out_path = "/home/viacheslav/Documents/9520_final/Datasets/"
    else:
        raise Exception("path not set up on machine")
    db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)

    training_layers = [1, 1, 1]
    genf_shape = [16, 16, 1]
    noise = None
    train_samples = 1000000
    test_samples = 1000000
    lrs = [1e-4,  1e-5, 2e-5, 4e-5, 8e-5, 1e-6, 2e-6, 4e-6, 8e-6, ]
    n_runs = 1
    n_iterations = 50000
    batch_size = 100
    fun_shape = [16, 16, 16, 1]
    em_shape = [256, 256,256]

    optimizer = "tf.train.GradientDescentOptimizer"
    scheduler = "none"




class cfg_bp_2_2:
        name = "cfg_bp_2_2"
        machine = socket.gethostname()
        if machine == "Ikarus":
            db_path = "/home/maksym/Desktop/slt/cfg_bpolynom"
            out_path = "/home/maksym/Desktop/slt/"
        elif machine.startswith("matlaber"):
            db_path = "/mas/u/mkkr/mk/slt/cfg_bpolynom"
            out_path = "/mas/u/mkkr/mk/slt/"
        elif machine == "viacheslav-HP-Pavilion-Notebook":
            db_path = "/home/viacheslav/Documents/9520_final/Datasets/cfg_bpolynom"
            out_path = "/home/viacheslav/Documents/9520_final/Datasets/"
        else:
            raise Exception("path not set up on machine")
        db_path = os.path.join(db_path)
        out_path = os.path.join(out_path, name)

        training_layers=[0,1,0]
        genf_shape = [784, 128, 10]
        noise = None
        train_samples = 100000
        test_samples = 100000
        lrs = [1e-4, 2e-4, 4e-4, 8e-4, 1e-5, 2e-5, 4e-5, 8e-5, 1e-6, 2e-6, 4e-6, 8e-6, ]
        n_runs = 5
        n_iterations = 50000
        batch_size = 100
        fun_shape = [784, 128, 64, 10]
        em_shape = [256,8192]

        optimizer="tf.train.GradientDescentOptimizer"
        scheduler="none"


class cfg_bp_2_3:
    name = "cfg_bp_2_3"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg_bpolynom"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg_spolynom"
        out_path = "/mas/u/mkkr/mk/slt/"
    elif machine == "viacheslav-HP-Pavilion-Notebook":
        db_path = "/home/viacheslav/Documents/9520_final/Datasets/cfg_bpolynom"
        out_path = "/home/viacheslav/Documents/9520_final/Datasets/"
    else:
        raise Exception("path not set up on machine")
    db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)

    training_layers = [0, 1, 0]
    genf_shape = [784, 128, 10]
    noise = None
    train_samples = 100000
    test_samples = 100000
    lrs = [1e-4, 2e-4, 4e-4, 8e-4, 1e-5, 2e-5, 4e-5, 8e-5, 1e-6, 2e-6, 4e-6, 8e-6, ]
    n_runs = 5
    n_iterations = 50000
    batch_size = 100
    fun_shape = [784, 128, 64, 10]
    em_shape = [256, 256,8192]

    optimizer = "tf.train.GradientDescentOptimizer"
    scheduler = "none"


class cfg_bp_12_2:
    name = "cfg_bp_12_2"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/bfg_spolynom"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg_bpolynom"
        out_path = "/mas/u/mkkr/mk/slt/"
    elif machine == "viacheslav-HP-Pavilion-Notebook":
        db_path = "/home/viacheslav/Documents/9520_final/Datasets/cfg_bpolynom"
        out_path = "/home/viacheslav/Documents/9520_final/Datasets/"
    else:
        raise Exception("path not set up on machine")
    db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)

    training_layers = [1, 1, 1]
    genf_shape = [784, 128, 10]
    noise = None
    train_samples = 100000
    test_samples = 100000
    lrs = [1e-4, 2e-4, 4e-4, 8e-4, 1e-5, 2e-5, 4e-5, 8e-5, 1e-6, 2e-6, 4e-6, 8e-6, ]
    n_runs = 5
    n_iterations = 50000
    batch_size = 100
    fun_shape = [784, 128, 64, 10]
    em_shape = [256, 8192]

    optimizer = "tf.train.GradientDescentOptimizer"
    scheduler = "none"


class cfg_bp_12_3:
    name = "cfg_bp_12_3"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg_bpolynom"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg_bpolynom"
        out_path = "/mas/u/mkkr/mk/slt/"
    elif machine == "viacheslav-HP-Pavilion-Notebook":
        db_path = "/home/viacheslav/Documents/9520_final/Datasets/cfg_bpolynom"
        out_path = "/home/viacheslav/Documents/9520_final/Datasets/"
    else:
        raise Exception("path not set up on machine")
    db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)

    training_layers = [1, 1, 1]
    genf_shape = [784, 128, 10]
    noise = None
    train_samples = 100000
    test_samples = 100000
    lrs = [1e-4, 2e-4, 4e-4, 8e-4, 1e-5, 2e-5, 4e-5, 8e-5, 1e-6, 2e-6, 4e-6, 8e-6, ]
    n_runs = 5
    n_iterations = 50000
    batch_size = 100
    fun_shape = [784, 128, 64, 10]
    em_shape = [256, 256,8192]

    optimizer = "tf.train.GradientDescentOptimizer"
    scheduler = "none"

class cfg_mnist_2_2:
        name = "cfg_mnist_2_2"
        machine = socket.gethostname()
        if machine == "Ikarus":
            db_path = "/home/maksym/Desktop/slt/cfg_mnist"
            out_path = "/home/maksym/Desktop/slt/"
        elif machine.startswith("matlaber"):
            db_path = "/mas/u/mkkr/mk/slt/cfg_mnist"
            out_path = "/mas/u/mkkr/mk/slt/"
        elif machine == "viacheslav-HP-Pavilion-Notebook":
            db_path = "/home/viacheslav/Documents/9520_final/Datasets/cfg_mnist"
            out_path = "/home/viacheslav/Documents/9520_final/Datasets/"
        else:
            raise Exception("path not set up on machine")
        db_path = os.path.join(db_path)
        out_path = os.path.join(out_path, name)

        training_layers = [0, 1, 0]
        genf_shape = [784, 128, 10]
        noise = None
        train_samples = 60000
        test_samples = 10000
        lrs = [2e-6]
        n_runs = 1
        n_iterations = 50000
        batch_size = 100
        fun_shape = [784, 128, 64, 10]
        em_shape = [1, 8192]

        optimizer = "tf.train.GradientDescentOptimizer"
        scheduler = "none"


class cfg_mnist_2_3:
    name = "cfg_mnist_2_3"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg_mnist"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg_mnist"
        out_path = "/mas/u/mkkr/mk/slt/"
    elif machine == "viacheslav-HP-Pavilion-Notebook":
        db_path = "/home/viacheslav/Documents/9520_final/Datasets/cfg_mnist"
        out_path = "/home/viacheslav/Documents/9520_final/Datasets/"
    else:
        raise Exception("path not set up on machine")
    db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)

    training_layers = [0, 1, 0]
    genf_shape = [784, 128, 10]
    noise = None
    train_samples = 60000
    test_samples = 10000
    lrs = [1e-4, 1e-5, 1e-6, 1e-7,1e-8, 1e-9]
    n_runs = 1
    n_iterations = 50000
    batch_size = 100
    fun_shape = [784, 128, 64, 10]
    em_shape = [16, 16, 8192]

    optimizer = "tf.train.GradientDescentOptimizer"
    scheduler = "none"


class cfg_mnist_12_2:
    name = "cfg_mnist_12_2"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg_mnist"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg_spolynom"
        out_path = "/mas/u/mkkr/mk/slt/"
    elif machine == "viacheslav-HP-Pavilion-Notebook":
        db_path = "/home/viacheslav/Documents/9520_final/Datasets/cfg_mnist"
        out_path = "/home/viacheslav/Documents/9520_final/Datasets/"
    else:
        raise Exception("path not set up on machine")
    db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)

    training_layers = [1, 1, 1]
    genf_shape = [784, 16, 10]
    noise = None
    train_samples = 60000
    test_samples = 10000
    lrs = [ 1.56e-7,7.8e-8,3.9e-8]
    n_runs = 1
    n_iterations = 50000
    batch_size = 100
    fun_shape = [784, 16, 16, 10]
    em_shape = [1, 256]

    optimizer = "tf.train.GradientDescentOptimizer"
    scheduler = "none"



class cfg_mnist_12:
    name = "cfg_mnist_12"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg_mnist"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg_spolynom"
        out_path = "/mas/u/mkkr/mk/slt/"
    elif machine == "viacheslav-HP-Pavilion-Notebook":
        db_path = "/home/viacheslav/Documents/9520_final/Datasets/cfg_mnist"
        out_path = "/home/viacheslav/Documents/9520_final/Datasets/"
    else:
        raise Exception("path not set up on machine")
    db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)

    training_layers = [1, 1, 1]
    genf_shape = [784,16, 10]
    noise = None
    train_samples = 60000
    test_samples = 10000
    lrs = [1.56e-7, 7.8e-8, 3.9e-8]
    n_runs = 1
    n_iterations = 1000
    batch_size = 100
    fun_shape = [784, 16, 16, 10]
    em_shape = [256, 256]

    optimizer = "tf.train.GradientDescentOptimizer"
    scheduler = "none"



class cfg_mnist_12_3:
    name = "cfg_mnist_12_3"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg_mnist"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg_mnist"
        out_path = "/mas/u/mkkr/mk/slt/"
    elif machine == "viacheslav-HP-Pavilion-Notebook":
        db_path = "/home/viacheslav/Documents/9520_final/Datasets/cfg_mnist"
        out_path = "/home/viacheslav/Documents/9520_final/Datasets/"
    else:
        raise Exception("path not set up on machine")
    db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)

    training_layers = [1, 1, 1]
    genf_shape = [784, 16, 10]
    noise = None
    train_samples = 60000
    test_samples = 10000
    lrs = [1.56e-7,7.8e-8,3.9e-8]
    n_runs = 1
    n_iterations = 1000000
    batch_size = 100
    fun_shape = [784, 16, 16, 10]
    em_shape = [256, 256, 256]

    optimizer = "tf.train.GradientDescentOptimizer"
    scheduler = "none"

class cfg_mnist_2:
    name = "cfg_mnist_2"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg_mnist"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg_spolynom"
        out_path = "/mas/u/mkkr/mk/slt/"
    elif machine == "viacheslav-HP-Pavilion-Notebook":
        db_path = "/home/viacheslav/Documents/9520_final/Datasets/cfg_mnist"
        out_path = "/home/viacheslav/Documents/9520_final/Datasets/"
    else:
        raise Exception("path not set up on machine")
    db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)

    training_layers = [1, 1, 1]
    genf_shape = [784,128, 10]
    noise = None
    train_samples = 60000
    test_samples = 10000
    lrs = [1.56e-7, 7.8e-8, 3.9e-8,0.1,0.2,0.3]
    n_runs = 1
    n_iterations = 100000
    batch_size = 100
    fun_shape = [784, 128, 64, 10]
    em_shape = [256, 8192]

    optimizer = "tf.train.GradientDescentOptimizer"
    scheduler = "none"



class cfg_mnist_3:
    name = "cfg_mnist_3"
    machine = socket.gethostname()
    if machine == "Ikarus":
        db_path = "/home/maksym/Desktop/slt/cfg_mnist"
        out_path = "/home/maksym/Desktop/slt/"
    elif machine.startswith("matlaber"):
        db_path = "/mas/u/mkkr/mk/slt/cfg_mnist"
        out_path = "/mas/u/mkkr/mk/slt/"
    elif machine == "viacheslav-HP-Pavilion-Notebook":
        db_path = "/home/viacheslav/Documents/9520_final/Datasets/cfg_mnist"
        out_path = "/home/viacheslav/Documents/9520_final/Datasets/"
    else:
        raise Exception("path not set up on machine")
    db_path = os.path.join(db_path)
    out_path = os.path.join(out_path, name)

    training_layers = [1, 1, 1]
    genf_shape = [784, 128, 10]
    noise = None
    train_samples = 60000
    test_samples = 10000
    lrs = [1.56e-7,7.8e-8,3.9e-8,0.1,0.2,0.3]
    n_runs = 1
    n_iterations = 10000
    batch_size = 100
    fun_shape = [784, 128, 64, 10]
    em_shape = [256, 256, 8192]

    optimizer = "tf.train.GradientDescentOptimizer"
    scheduler = "none"