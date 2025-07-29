#!/bin/bash

#SBATCH --job-name="cifar10-multi-gpu"
#SBATCH --output="output/%j.out"
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --gpu-bind=closest
#SBATCH --account=becs-delta-gpu
#SBATCH --time=00:02:00
# #SBATCH --mail-user=<user-email>
# #SBATCH --mail-type=BEGIN,FAIL,END

# if using conda, create a conda environment and install necessary packages
# see delta docs on how to do so
#
# module load anaconda3_gpu
# conda activate <myenv>

srun python -u cifar10-fabric-tb.py \
     --accelerator gpu \
     --devices 2 \
     --num_nodes 1 \
     --stratedy ddp \
     --precision "bf16-mixed" \
     --num_workers 1 \
     --lr 1e-3 \
     --epoch 10 # \
    # --resume-from-checkpoint "checkpoint_4.ckpt" # make sure that the checkpoint exists
     
