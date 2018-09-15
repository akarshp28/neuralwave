#! /bin/bash

# ===== PBS OPTIONS =====
### Set the job name
#PBS -N data_prep

### Specify queue to run in
#PBS -q copperhead

### Specify number of CPUs for job
#PBS -l nodes=1:ppn=16,mem=32GB

# ==== Main ======
INPUTFILE="/users/kjakkala/neuralwave/preprocess/preprocess_level1/preprocess_level1.m"

# ===== END APP OPTIONS =====

### run job
module load matlab/R2018a
matlab -nodisplay -nodesktop -nojvm -nosplash < $INPUTFILE

