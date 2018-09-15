#! /bin/bash

# ===== PBS OPTIONS =====
### Set the job name
#PBS -N data_prep

### Specify queue to run in
#PBS -q copperhead

### Specify number of CPUs for job
#PBS -l nodes=1:ppn=1,mem=128GB

# ==== load modules ======
module load anaconda3/5.0.1

# ==== Main ======
python /users/kjakkala/neuralwave/preprocess/pca.py -s $1 -d $2 -f $3
