#! /bin/bash

# ===== PBS OPTIONS =====
### Set the job name
#PBS -N data_prep

### Specify queue to run in
#PBS -q copperhead

### Specify number of CPUs for job
#PBS -l nodes=1:ppn=1,mem=128GB

#PBS -m bea
#PBS -M kjakkala@uncc.edu

# ==== load modules ======
module load anaconda3/5.0.1

# ==== Main ======
python /users/kjakkala/neuralwave/preprocess/preprocess_l2.py --src $1 --dst $2 --file $3 --sampling $4 --cols $5 --pca $6
