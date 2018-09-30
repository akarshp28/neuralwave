#!/usr/bin/env python3

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.io import loadmat
import numpy as np
import argparse
import pickle
import time
import h5py
import math
import sys
import os

def get_csi(x):
    x = np.squeeze(x["csi_trace"])
    data = []
    for i in range(x.shape[0]):
        array = {}
        array["timestamp_low"] = np.squeeze(x[i][0][0][0])
        array["bfee_count"] =  np.squeeze(x[i][0][0][1])
        array["Nrx"] = np.squeeze(x[i][0][0][2])
        array["Ntx"] = np.squeeze(x[i][0][0][3])
        array["rssi_a"] = np.squeeze(x[i][0][0][4])
        array["rssi_b"] = np.squeeze(x[i][0][0][5])
        array["rssi_c"] = np.squeeze(x[i][0][0][6])
        array["noise"] = np.squeeze(x[i][0][0][7])
        array["agc"] = np.squeeze(x[i][0][0][8])
        array["perm"] = np.squeeze(x[i][0][0][9])
        array["rate"] = np.squeeze(x[i][0][0][10])
        array["csi"] = np.squeeze(x[i][0][0][11])
        
        data.append(array)
    return data

def phase_correction(ph_raw):
    m = np.arange(-28,29)
    Tp = np.unwrap(ph_raw)
    k_param = (Tp[29] - Tp[0])/(m[29] - m[0]);
    b_param = np.sum(Tp)*(1/30)

    correct_phase = []
    for i in range(30):
        correct_phase.append(Tp[i] - k_param*m[i] - b_param)
    return correct_phase

# 3 x 3 MIMO Matrix format
# [h11 h12 h13
# h21 h22 h23
# h31 h32 h33]
def apply_phcorrect(ph_raw):
    mimo_mat = np.rollaxis(ph_raw, 2, 0)
    mimo_mat = np.reshape(mimo_mat, (30, 9))

    crct_ph = []
    for col in range(9):
        crct_ph.append(phase_correction(np.array(mimo_mat)[:, col]))

    stack_crc_ph = np.vstack(crct_ph).T

    restore_ph_mat = []
    for i in range(30):
        restore_ph_mat.append(stack_crc_ph[i, :].reshape((3,3)))
    return np.array(restore_ph_mat).T

def fill_gaps(csi_trace, technique):
    amp_data = []
    ph_data = []

    for ind in range(len(csi_trace)):
        csi_entry = csi_trace[ind]

        scaled_csi = get_scaled_csi(csi_entry)
        amp = np.absolute(scaled_csi)
        ph = np.angle(scaled_csi)

        amp_temp=[]
        ph_temp=[]

        if technique == 'fill':
            if csi_trace[ind]['Ntx'] == 1:
                ph = np.expand_dims(ph, axis=0)
                amp = np.expand_dims(amp, axis=0)
                for i in range(30):
                    amp_temp.append(np.append(amp[:,:,i], np.zeros((2,3)) + np.nan).reshape((3,3)))
                    ph_temp.append(np.append(ph[:,:,i], np.zeros((2,3)) + np.nan).reshape((3,3)))
                amp_data.append(np.array(amp_temp).flatten())
                ph_data.append(apply_phcorrect(ph_temp).flatten())

            elif csi_trace[ind]['Ntx'] == 2:
                for i in range(30):
                    amp_temp.append(np.append(amp[:,:,i], np.zeros((1,3)) + np.nan).reshape((3,3)))
                    ph_temp.append(np.append(ph[:,:,i], np.zeros((1,3)) + np.nan).reshape((3,3)))
                amp_data.append(np.array(amp_temp).flatten())
                ph_data.append(apply_phcorrect(ph_temp).flatten())

            elif csi_trace[ind]['Ntx'] == 3:
                amp_data.append(np.array(amp).T.flatten())
                ph_data.append(apply_phcorrect(ph).T.flatten())

        elif technique == 'mean':
            if csi_trace[ind]['Ntx'] == 1:
                ph = np.expand_dims(ph, axis=0)
                amp = np.expand_dims(amp, axis=0)

                mean_amp = np.mean(amp)
                mean_ph = np.mean(ph)

                for i in range(30):
                    amp_temp.append(np.append(amp[:,:,i], np.zeros((2,3)) + mean_amp).reshape((3,3)))
                    ph_temp.append(np.append(ph[:,:,i], np.zeros((2,3)) + mean_ph).reshape((3,3)))
                ph_temp = np.array(ph_temp).T
                amp_data.append(np.array(amp_temp).flatten())
                ph_data.append(apply_phcorrect(ph_temp).flatten())

            elif csi_trace[ind]['Ntx'] == 2:
                mean_amp = np.mean(amp)
                mean_ph = np.mean(ph)
                for i in range(30):
                    amp_temp.append(np.append(amp[:,:,i], np.zeros((1,3)) + mean_amp).reshape((3,3)))
                    ph_temp.append(np.append(ph[:,:,i], np.zeros((1,3)) + mean_ph).reshape((3,3)))
                ph_temp = np.array(ph_temp).T
                amp_data.append(np.array(amp_temp).flatten())
                ph_data.append(apply_phcorrect(ph_temp).flatten())

            elif csi_trace[ind]['Ntx'] == 3:
                amp_data.append(np.array(amp).T.flatten())
                ph_data.append(apply_phcorrect(ph).T.flatten())

    return np.hstack([amp_data, ph_data])

def dbinv(x):
    return np.power(10, (np.array(x)/10))

def get_total_rss(csi_st):
    rssi_mag = 0;
    if csi_st['rssi_a'] != 0:
        rssi_mag = rssi_mag + dbinv(csi_st['rssi_a'])

    if csi_st['rssi_b'] != 0:
        rssi_mag = rssi_mag + dbinv(csi_st['rssi_b'])

    if csi_st['rssi_c'] != 0:
        rssi_mag = rssi_mag + dbinv(csi_st['rssi_c'])

    return 10*np.log10(rssi_mag) - 44 - csi_st['agc']

def get_scaled_csi(csi_st):
    csi = csi_st['csi']

    csi_sq = np.multiply(csi, np.conj(csi))
    csi_pwr = np.sum(csi_sq[:])
    rssi_pwr = dbinv(get_total_rss(csi_st))

    scale = rssi_pwr / (csi_pwr / 30)

    if (csi_st['noise'] == -127):
        noise_db = -92
    else:
        noise_db = csi_st['noise']

    thermal_noise_pwr = dbinv(noise_db)
    quant_error_pwr = scale * (csi_st['Nrx'] * csi_st['Ntx'])
    total_noise_pwr = thermal_noise_pwr + quant_error_pwr;

    ret = csi * np.sqrt(scale / total_noise_pwr);
    if csi_st['Ntx'] == 2:
        ret = ret * np.sqrt(2);
    elif csi_st['Ntx'] == 3:
        ret = ret * np.sqrt(dbinv(4.5));

    return ret

def read_samples(dataset_path, endswith=".csv"):
    datapaths, labels = list(), list()
    label = 0
    classes = sorted(os.walk(dataset_path).__next__()[1])

    # List each sub-directory (the classes)
    for c in classes:
        c_dir = os.path.join(dataset_path, c)
        walk = os.walk(c_dir).__next__()
        # Add each image to the training set
        for sample in walk[2]:
            # Only keeps csv samples
            if sample.endswith(endswith):
                datapaths.append(os.path.join(c_dir, sample))
                labels.append(label)
        label += 1

    return datapaths, labels

def smooth(x,window_len):
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    w=np.hanning(window_len)
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(window_len//2):-(window_len//2)]

def compute_data(file_path, sampling, cols1, cols2):
    if (not os.path.isfile(file_path)):
        raise ValueError("File dosn't exits")

    csi_trace = get_csi(loadmat(file_path))[2000:10000]
    csi_trace = csi_trace[::sampling]
    csi_trace = fill_gaps(csi_trace, technique='mean')[:, cols1:cols2]

    return csi_trace.astype(np.float32)

#******************************************************************************#

#sampling 1: 8000
#sampling 2: 4000
#sampling 4: 2000
#sampling 8: 1000
#sampling 16: 500

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="source dir")
ap.add_argument("--dst", required=True, help="destination dir")
ap.add_argument("--file", required=True, help="filename without extension for scalers and dataset")
ap.add_argument("--sampling", required=True, type=int, help="sampling rate for data")
ap.add_argument("--cols", required=True, help="Data that needs to be computed AMP/PH/ALL")
ap.add_argument("--pca", required=True, type=float, help="percent of variance that needs to be explained by PCA (0-1)")
args = vars(ap.parse_args())

src_path = args["src"]
dest_path = args["dst"]
dest_file = args["file"]
sampling = args["sampling"]
cols = args["cols"].strip()
pca_var = args["pca"]

if cols == "AMP":
        cols1 = 0
        cols2 = 270
        cols = 270
elif cols == "PH":
        cols1 = 270
        cols2 = 540
        cols = 270
elif cols == "ALL":
        cols1 = 0
        cols2 = 540
        cols = 540
else:
        raise ValueError("Check cols argument!! Got: {} | Acceptable arguments (AMP/PH/ALL)".format(cols))

filter_size = 91
rows = int(8000/sampling)

print("rows: ", rows, "cols: ", cols1, "-", cols2, "PCA Var: ", pca_var)
sys.stdout.flush()

files, labels = read_samples(src_path, ".mat")

dset_X = []
dset_y = []

for i in range(len(files)):
    tmp = compute_data(files[i], sampling, cols1, cols2)
    if (tmp.shape == (rows, cols)):
        for j in range(cols):
            tmp[:, j] = smooth(tmp[:, j], filter_size)
        dset_X.append(tmp)
        dset_y.append(labels[i])
    else:
        print("File dimension error | File:{} | Size:{}", files[i], tmp.shape)

dset_X = np.array(dset_X)
dset_y = np.array(dset_y)

print("Data shape: ", dset_y.shape, dset_X.shape)
sys.stdout.flush()

train_X, test_X, train_y, test_y = train_test_split(dset_X, dset_y, test_size=0.15, random_state=42)

means = np.mean(np.mean(train_X, axis=0), axis=0)
train_X -= means

mins = np.min(np.min(train_X, axis=0), axis=0)
maxs = np.max(np.max(train_X, axis=0), axis=0)
train_X -= mins
train_X /= (maxs-mins)

test_X -= means
test_X -= mins
test_X /= (maxs-mins)

pca = PCA(n_components=pca_var)
train_X = pca.fit_transform(train_X.reshape((train_X.shape[0], -1)))
test_X = pca.transform(test_X.reshape((test_X.shape[0], -1)))

hf = h5py.File(os.path.join(dest_path, dest_file+'.h5'), 'w')
hf.create_dataset('X_train', data=train_X)
hf.create_dataset('y_train', data=train_y)
hf.create_dataset('X_test', data=test_X)
hf.create_dataset('y_test', data=test_y)
hf.close()

dict = {'pca': pca, 'means': means, 'mins': mins, 'maxs': maxs}
fileObject = open(os.path.join(dest_path, dest_file+'_scalers'+'.pkl'),'wb')
pickle.dump(dict, fileObject)
fileObject.close()

print("finished!!")

