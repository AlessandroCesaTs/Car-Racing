#!/bin/bash
#SBATCH --no-requeue
#SBATCH -J drive
#SBATCH --get-user-env
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=#1
#SBATCH -o drive.out
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00

source environment/bin/activate

for time_horizon in 15
do
    echo "started time horizon $time_horizon"
    python -u source/drive.py --time_horizon $time_horizon
    echo "done time horizon $time_horizon"
done

echo "done"
