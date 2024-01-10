#!/bin/bash
#SBATCH --job-name=rebar_seg   # create a short name for your job
#SBATCH --partition=GPU              # specify the GPU partition
#SBATCH --gres=gpu:1                 # request two GPU
#SBATCH --nodes=1                    # node count
#SBATCH --ntasks=1                   # total number of tasks across all nodes
#SBATCH --cpus-per-task=128            # cpu-cores per task 
#SBATCH --mem=640G                    # memory per cpu-core 
#SBATCH --time=02-00:00:00              # total run time limit (HH:MM:SS)
#SBATCH --output=%j.%x.out           # file to collect standard output
#SBATCH --error=%j.%x.err            # file to collect standard error
#SBATCH --mail-type=begin            # send email when job begins
#SBATCH --mail-type=end              # send email when job ends
#SBATCH --mail-user=u6771897@anu.edu.au

module load mamba
# Start Virtual Environment
source activate myenv

# Get All Dependencies
# pip install -r requirments.txt


# Start Training Model 
python train.py --multi-threading