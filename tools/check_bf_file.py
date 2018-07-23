#!/usr/bin/env python3
from numpy.ctypeslib import ndpointer
from ctypes import *
import numpy as np
import os.path
import sys

#ctypes module to call c++ functions
lib = cdll.LoadLibrary('/home/kalvik/Documents/neuralwave/preprocess/libbfee.so')

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

def main():
    if (len(sys.argv) != 2):
        raise Exception("Usage: check_bf_file.py <dat file>")

    if (not os.path.isfile(sys.argv[1])):
        raise Exception("file doesn't exist: {}".format(sys.argv[1]))

    print("File has {} packets".format(len(read_bf_file(sys.argv[1]))))

if __name__ == "__main__":
    main()
