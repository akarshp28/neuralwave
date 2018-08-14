#! /bin/bash

# ===== PBS OPTIONS =====
### Set the job name
#PBS -N data_prep

### Specify queue to run in
#PBS -q copperhead

### Specify number of CPUs for job
#PBS -l nodes=1:ppn=1,mem=1GB

# mail alert at start, end and abortion of execution
#PBS -m bea

# send mail to this address
#PBS -M kjakkala@uncc.edu

# ==== load modules ======
module load anaconda3/5.0.1

# ==== Main ======
dst_root="/scratch/kjakkala/neuralwave/data/preprocess_level3/train/"
src_root="/scratch/kjakkala/neuralwave/data/preprocess_level2/train/"

python3 /scratch/kjakkala/neuralwave/preprocess/to_tfrecord.py -s "$src_root" -d "$dst_root" -l "$PBS_ARRAYID"

