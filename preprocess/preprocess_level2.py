import os
import sys
import math
import h5py
import numpy as np
from mpi4py import MPI
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def smooth(x,window_len):
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    w=np.hanning(window_len)
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(window_len//2):-(window_len//2)]

def read_samples(dataset_path, endswith=".csv"):
    datapaths, labels = list(), list()
    label = 0
    classes = sorted(os.listdir(dataset_path))
    # List each sub-directory (the classes)
    for c in classes:
        c_dir = os.path.join(dataset_path, c)
        walk = os.listdir(c_dir)
        # Add each image to the training set
        for sample in walk:
            # Only keeps csv samples
            if sample.endswith(endswith):
                datapaths.append(os.path.join(c_dir, sample))
                labels.append(label)
        label += 1
    return np.array(datapaths), np.array(labels), classes

def read_array(data_path):
    return np.loadtxt(open(data_path, "rb"), delimiter=",", dtype=np.float32)

def process_sample(data_path, dest_path, min_, max_, scalers):
    data = read_array(data_path)

    for i in range(cols):
        data[:, i] = scalers[i].transform(np.expand_dims(data[:, i], axis=0))
    data = (data - min_)/(max_ - min_)

    for i in range(cols):
        data[:, i] = smooth(data[:, i], 91)

    if (data.shape != (rows, cols)):
        print(data.shape, data_path)
        sys.stdout.flush()
        return

    path, file = os.path.split(data_path)
    _, class_name = os.path.split(path)

    np.savetxt((os.path.join(os.path.join(dest_path, class_name), file)), data.astype(np.float32), delimiter=",")

#******************************************************************************#

src_path = "/users/kjakkala/neuralwave/data/preprocess_level1"
dest_path = "/users/kjakkala/neuralwave/data/preprocess_level2"
rows = 8000
cols = 540

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

X, y, classes = read_samples(src_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

num_sl = int(math.floor(len(X_train)/size))
train_sl = [None for _ in range(num_sl)]
train_array = None

data_c_last = None
last_sl = None
data_c_len = int(math.floor(len(X))/size)
data_c = [None for _ in range(data_c_len)]

if (rank == 0):
    print("train size:", len(X_train), "test size:", len(X_test))
    sys.stdout.flush()

    train_sl = [X_train[i:i+size] for i in range(0, len(X_train), size)]
    train_array = np.empty((size, rows, cols), dtype=np.float32)

    if (len(train_sl[-1]) != size):
        last_sl = train_sl[-1]
        del train_sl[-1]

    scalers = [StandardScaler(with_std = False) for _ in range(cols)]

    min_ = float('Inf')
    max_ = -float('Inf')

    print("Started calculating scalers")
    sys.stdout.flush()

for index in range(num_sl):
    addr = comm.scatter(train_sl[index], root=0)
    comm.Gatherv(np.expand_dims(read_array(addr), axis=0), train_array, root=0)

    if (rank == 0):
        for i in range(cols):
            scalers[i].partial_fit(train_array[:, :, i])

        min_temp = np.min(train_array)
        max_temp = np.max(train_array)

        min_ = min(min_temp, min_)
        max_ = max(max_temp, max_)

        sys.stdout.write("\r{}/{}".format(index+1, num_sl))
        sys.stdout.flush()

if (rank == 0):
    if isinstance(last_sl, (list,)):
        train_array = np.array([read_array(addr) for addr in last_sl])

        for i in range(cols):
            scalers[i].partial_fit(train_array[:, :, i])

        min_temp = np.min(train_array)
        max_temp = np.max(train_array)

        min_ = min(min_temp, min_)
        max_ = max(max_temp, max_)

    hf = h5py.File('/users/kjakkala/neuralwave/data/scalers.h5', 'w')
    hf.create_dataset('min', data=min_)
    hf.create_dataset('max', data=max_)
    hf.create_dataset('scalers', data=scalers)
    hf.close()

    if not os.path.exists(os.path.join(dest_path, "train")):
        os.mkdir(os.path.join(dest_path, "train"))
        for i in classes:
            os.mkdir(os.path.join(os.path.join(dest_path, "train"), i))

    if not os.path.exists(os.path.join(dest_path, "test")):
        os.mkdir(os.path.join(dest_path, "test"))
        for i in classes:
            os.mkdir(os.path.join(os.path.join(dest_path, "test"), i))

    data_tmp = [[X_train[i], os.path.join(dest_path, "train"), min_, max_, scalers] for i in range(len(X_train))]
    data_tmp.extend([X_test[i], os.path.join(dest_path, "test"), min_, max_, scalers] for i in range(len(X_test)))
    data_c = [data_tmp[i:i+size] for i in range(0, len(data_tmp), size)]

    if (len(data_c[-1]) < size):
        data_c_last = data_c[-1]
        del data_c[-1]

    print("Started writing csv files")
    sys.stdout.flush()

for index in range(data_c_len):
    data_tmp = comm.scatter(data_c[index], root=0)
    process_sample(data_tmp[0], data_tmp[1], data_tmp[2], data_tmp[3], data_tmp[4])

    if (rank == 0):
        sys.stdout.write("\r{}/{}".format(index+1, data_c_len))
        sys.stdout.flush()

if (rank == 0):
    if isinstance(data_c_last, (list,)):
        for data_tmp in data_c_last:
            process_sample(data_tmp[0], data_tmp[1], data_tmp[2], data_tmp[3], data_tmp[4])

print("\nFinished !!")
sys.stdout.flush()
