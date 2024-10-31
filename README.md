# ODE-SDE gap repository

## Overview
This repo contains the code implementation used to generate the results in the paper: "Closing the ODE-SDE gap in score-based diffusion models through the Fokker-Planck equation".

## Supplied models and data
Checkpoints for all trained models presented in the paper are given in the ```checkpoints``` directory. Weights of the discretised distributions associated with samples from each model are saved in the ```precomputed_dists``` directory. To ensure the accuracy of the numerics, this work uses a high number of epochs and large sample sizes, so these pre-supplied models are included to avoid a need to resample them. Alternativeley, scripts to train these models or generate these distributions are outlined below. 

## To generate the results
The ```experiment_plotting_precomputed.py``` script can be used to generate the main figure in the paper using the precomputed weights supplied. This also prints all Wasserstein distances to the screen that can be used to populate the tables and supplementary figures.

To reproduce these weights using the trained models the ```experiment_plotting.py``` script can be used. This samples points from each model and saves the associated weights to .npy files, before plotting the corresponding results.

To make an explicit comparison for a single case, the ```query_model.py``` script can be used. This script allows you to specify a sample size and a model to sample from, and will return figures visualising the ODE and SDE sample distributions, as well as printing Wasserstein distances to the screen.

Moreover, if you wish to also retrain the models from scratch, the ```OUPINN_loop.py``` script will train and store models for all data distributions and regularisation weights. A single model can also be trained using the ```OUPINN.py``` script. The progress of training can be monitored through tensorboard and tensorboard logs are saved in the ```summaries``` directory.

For reproducing the weights or retraining the models a GPU is highly recommended. In the case of training this can take several hours.

To generate the plot of the data distributions the ```data_dist_plot.py``` script can be used. 

All other files (```data_dists.py```,```diffusion_utils.py```, ```models.py```, ```pinn_utils.py```) contain utility functions and classes used to implement the algorithms.