#!/bin/bash
#SBATCH --no-requeue
#SBATCH -J learn_system
#SBATCH --get-user-env
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=#1
#SBATCH -o learn_system.out
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00

source environment/bin/activate


for lr in 0.001
do
    echo "started learning rate $lr"
    python -u source/learn_system.py --learning_rate $lr --epochs 100
    echo "done learning rate $lr"
done

echo "done"
