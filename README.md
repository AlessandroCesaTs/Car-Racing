# Car Racing
This repository contains the report and the code of the project of Alessandro Cesa for the exam in Cyber Physical Systems Project and Pattern Recognition by prof. Laura Nenzi, in the academic year 2023/24

The project consists in controlling with Model Predictive Control the car of the "Car Racing" environment from Gym, using an Auto Encoder deep neural Network.

## Repository Structure

The script `generate_data.sh` is used to generate the dataset on which the neural network will be trained; `learn_system.sh` is used for training the network, and `drive.sh` is used to run the simulation with the MPC controller.

In the folder `source` there is all the code needed for the project.
In `models` there are the trained models, and in `plots` the plots that are discussed in the report.

## Reproducing the results

In order to reproduce the results you need to at first clone the repository and install the required packages with `pip install -r requirements.txt`.

Now, if you just want to run the simulation, run `drive.sh`. The script is intended for running with the command `sbatch` in a cluster with the slurm job scheduler; otherwise you can just delete the lines starting with `#SBATCH` and run it like a normal Shell script.

If you want to train the model from scratch, you'll fist have to generate the dataset running `generate_data.sh` and then train the model running `learn_system.sh` (as before, modify the scripts if you're not using slurm).
