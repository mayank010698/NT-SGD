#!/usr/bin/env bash

#SBATCH --time=24:00:00
#SBATCH --job-name=exp
#SBATCH --partition=dali
#SBATCH --nodes=1
#SBATCH --gpus=1
. /projects/dali/spack/share/spack/setup-env.sh && spack env activate dali



for config in $(cat all_json)
do
    python main.py $config
done

