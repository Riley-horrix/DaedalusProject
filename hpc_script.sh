#!/bin/bash

#PBS -lwalltime=2:00:00
#PBS -lselect=1:ncpus=1:mem=16gb:ngpus=1

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

python -m src.scripts.runner --wandb