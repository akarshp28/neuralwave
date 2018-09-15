#! /bin/bash

# ===== PBS OPTIONS =====
### Set the job name
#PBS -N classifier

### Specify queue to run in
#PBS -q copperhead

### Specify number of CPUs for job
#PBS -l nodes=1:ppn=1:gpus=1,mem=16GB

# mail alert at start, end and abortion of execution
#PBS -m bea

# send mail to this address
#PBS -M kjakkala@uncc.edu

# ==== load modules ======
module load tensorflow/1.7.0-anaconda3-cuda9

# ==== Main ======
python3 /scratch/kjakkala/neuralwave/classifiers/resnet25_mc.py
