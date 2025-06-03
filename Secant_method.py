#!/home/hd/hd_hd/hd_ar307/.conda/envs/env_0/bin/python3

#SBATCH --job-name=Secant_method_CL
#SBATCH --output=Secant_method_%A.out

#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=10gb



import numpy as np
import os
import math
import threading

import sys


on = "Helix"

if on == "Helix": 
    # Aggiungi il percorso alla variabile sys.path
    home = "/home/hd/hd_hd/hd_ar307/finale/"
    sys.path.append(home)

    from mylib import *

    c = "A100"
    Run_command="sbatch --gres=gpu:"+c+":1 --array=" 

    print("Running on "+on +" with " +c+"\n")

if on == "EINC":
    # Aggiungi il percorso alla variabile sys.path
    home = "/wang/users/luca99/cluster_home/2d-dipolar/"
    sys.path.append(home)

    from mylib import *

    card={ "3090":"rtx3090", "A100": "a100", "splitted": "1g.10gb"}
    c = card["A100"]

    Run_command = "sbatch jobscript.sh --gres=gpu:"+c+":1 --array=" 
    #"nohup srun -p einc -A einc --gres=gpu:"+c+":1 singularity exec --app ido /containers/testing/f27_c21380p5_2023-10-16_2.img ./2D_bosegascl"

    print("Running on "+on +" with " +c+"\n")


LAT=
m = 0.5
T = 1.25
g_2d = 0.1
eps_dd = 0
sigma = 0

offset=0
N_Runs = 3



mu_delta = 0.01
mu_guess = 0.1

t0_percentage = 1./6


Secant_method_function( on, 
                        home, 
                        LAT, T, g_2d, eps_dd, sigma, 
                        mu_guess, mu_delta, 
                        t0_percentage, N_Runs, offset, 
                        "Makefile", "main.cu", "Parameters.h", 
                        c, Run_command, 
                        plot=True)







