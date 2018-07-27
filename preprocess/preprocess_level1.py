#!/usr/bin/env python3
from numpy.ctypeslib import ndpointer
import multiprocessing as mp
from ctypes import *
import numpy as np
import argparse
import time
import sys
import os

#ctypes module to call c++ functions
lib = cdll.LoadLibrary('/users/kjakkala/neuralwave/preprocess/libbfee.so')

# Instantiate the parser
parser = argparse.ArgumentParser()
# Optional positional argument
parser.add_argument('--src',
                    help='path of raw .dat files folder',
                    required=False,
                    default="/home/kalvik/shared/CSI_DATA/raw/")
# Optional argument
parser.add_argument('--dest',
                    help='path of destination folder',
                    required=False,
                    default="/home/kalvik/shared/CSI_DATA/preprocessed_level1/")
# Optional argument
parser.add_argument('--jobs',
                    type=int,
                    help='number of jobs for parallization',
                    required=False,
                    default=16)

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

    return np.array(amp_data), np.array(ph_data)

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

#call the computational routine in c++
def read_bfee(byts):
    obj = lib.bfee_c()
    lib.read_bfee_c(obj, byts)

    array = {}
    array["timestamp_low"] = c_uint(lib.get_timestamp_low(obj)).value
    array["bfee_count"] = c_short(lib.get_bfee_count(obj)).value
    array["Nrx"] = c_uint(lib.get_Nrx(obj)).value
    array["Ntx"] = c_uint(lib.get_Ntx(obj)).value
    array["rssi_a"] = c_uint(lib.get_rssi_a(obj)).value
    array["rssi_b"] = c_uint(lib.get_rssi_b(obj)).value
    array["rssi_c"] = c_uint(lib.get_rssi_c(obj)).value
    array["noise"] = c_byte(lib.get_noise(obj)).value
    array["agc"] = c_uint(lib.get_agc(obj)).value
    array["rate"] = c_uint(lib.get_rate(obj)).value

    lib.get_perm.restype = ndpointer(dtype=c_int, shape=(3,))
    array["perm"] = lib.get_perm(obj)

    lib.get_csi.restype = ndpointer(dtype=np.complex128, shape=(array["Ntx"], array["Nrx"], 30))
    array["csi"] = np.flip(lib.get_csi(obj), 2)

    lib.del_obj(obj)

    return array

def read_bf_file(filename):
    f = open(filename, "rb")

    #get length of file
    f.seek(0, 2)
    length = f.tell()
    f.seek(0, 0)

    # Initialize variables
    ret = []               #Holds the return values - 1x1 CSI is 95 bytes big, so this should be upper bound
    cur = 0                #Current offset into file
    count = 0              #Number of records output
    broken_perm = 0        #Flag marking whether weve encountered a broken CSI yet
    triangle = [0, 2, 5]   #What perm should sum to for 1,2,3 antennas

    # Process all entries in file
    # Need 3 bytes -- 2 byte size field and 1 byte code
    while (cur < (length-3)):
        #Read size and code
        field_len = int.from_bytes(f.read(2), byteorder='big')
        code = int.from_bytes(f.read(1), byteorder='little')
        cur += 3

        #If unhandled code, skip (seek over) the record and continue
        if(code == 187): # get beamforming or phy data
            byts = f.read(field_len-1)
            cur += field_len-1
            if(len(byts) != (field_len-1)):
                f.close()
                return ret[:count]
        else: #skip all other info
            f.seek(field_len-1, 1)
            cur += field_len-1
            continue

        if (code == 187): #hex2dec('bb')) Beamforming matrix -- output a record
            count += 1
            ret.append(read_bfee(byts))
            perm = ret[count-1]["perm"]
            Nrx = ret[count-1]["Nrx"]

            if (Nrx == 1): # No permuting needed for only 1 antenna
                continue

            if (sum(perm) != triangle[Nrx-1]): # matrix does not contain default values
                if (broken_perm == 0):
                    broken_perm = 1
                    #print('WARN ONCE: Found CSI ({}) with Nrx={} and invalid perm={}'.format(filename, Nrx, perm))

            else:
                ret[count-1]["csi"][:, perm[:Nrx-1], :] = ret[count-1]["csi"][:, :Nrx-1, :]

    # Close file
    f.close()

    return ret[:count]

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

def compute_data(file_path, class_sample_index):
    csi_trace = read_bf_file(file_path)[2000:10000]
    X_amp, X_ph = fill_gaps(csi_trace, technique='mean')

    if ((X_amp.shape != (8000, 270)) or (X_ph.shape != (8000, 270))):
        print(X_amp.shape, X_ph.shape, file_path)
        return

    path, file = os.path.split(file_path)
    _, class_name = os.path.split(path)

    np.savetxt(os.path.join(os.path.join(dest_path, class_name), "{}{}.csv".format(class_name, str(class_sample_index))), np.concatenate((X_amp, X_ph), axis=-1), delimiter=",")

def main(src_path, dest_path, jobs):
    x, y = read_samples(src_path, ".dat")
    y_ = []

    classes = os.walk(src_path).__next__()[1]
    for i in range(len(classes)):
        y_.extend(list(range(y.count(i))))

    for class_name in classes:
        if not os.path.exists(os.path.join(dest_path, class_name)):
            os.makedirs(os.path.join(dest_path, class_name))

    procs = []

    for i in range(len(x)):
        proc = Process(target=compute_data, args=(x[i], y_[i]))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.src, args.dest, args.jobs)
