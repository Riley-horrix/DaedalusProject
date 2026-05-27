#!/bin/bash

#PBS -lwalltime=70:00:00
#PBS -lselect=1:ncpus=1:mem=32gb:ngpus=1

#PBS -o ./logs/hpc/my_job.out
#PBS -e ./logs/hpc/my_job.err

set -e

cd $PBS_O_WORKDIR

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate pytorch_env

# ls
cmd="ls --almost-all --color=auto --classify --group-directories-first --human-readable -l --literal --show-control-chars --tabsize=0"
echo $(date +"%Y-%m-%d %H:%M:%S") runprogram: $cmd
eval $cmd

# nvidia-smi
cmd="nvidia-smi"
echo $(date +"%Y-%m-%d %H:%M:%S") runprogram: $cmd
eval $cmd

export WANDB_API_KEY=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

python -m src.scripts.runner_layered --wandb --data_path=./data/
