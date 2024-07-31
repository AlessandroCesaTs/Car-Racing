#!/bin/bash
#SBATCH --no-requeue
#SBATCH -J generate_data
#SBATCH --get-user-env
#SBATCH --partition=THIN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o generate_data.out
#SBATCH --mem=0
#SBATCH --time=01:00:00

source environment/bin/activate

python -u source/data_generator.py --num_of_transitions 40000

echo "done"
