#!/bin/bash

#SBATCH --job-name=ann_bench_puffinn# Job name
#SBATCH --output=outs/%x.out      # Name of output file (%j expands to jobId)
#SBATCH --error=outs/%x.err 
#SBATCH --partition=brown    # Run on either the Red or Brown queue
#SBATCH -c 4        # Schedule 4 cores
#SBATCH --time=05:00:00          # Run time (hh:mm:ss) - run for one hour max
#SBATCH --mem-per-cpu=8G # memo

cd $HOME/repositories/ann-benchmarks
module load Anaconda3
. $(conda info --base)/etc/profile.d/conda.sh
conda activate annb
python run.py --local --runs 2 -k 10 --parallel 4


