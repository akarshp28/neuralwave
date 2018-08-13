#! /bin/bash

# ===== PBS OPTIONS =====
### Set the job name
#PBS -N data_prep

### Specify queue to run in
#PBS -q copperhead

### Specify number of CPUs for job
#PBS -l nodes=1:ppn=32

# ==== load modules ======
module load openmpi/1.10.0
module load anaconda3/5.0.1

# ==== Main ======
mpirun -n 32 python /users/kjakkala/neuralwave/preprocess/preprocess_level2.py
