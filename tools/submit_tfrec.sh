#! /bin/bash

# ===== PBS OPTIONS =====
### Set the job name
#PBS -N data_prep

### Specify queue to run in
#PBS -q copperhead

### Specify number of CPUs for job
#PBS -l nodes=1:ppn=1:gpus=1,mem=1GB

# ==== load modules ======
module load tensorflow/1.7.0-anaconda3-cuda9

# ==== Main ======
python3 /users/kjakkala/neuralwave/classifiers/resnet25.py
