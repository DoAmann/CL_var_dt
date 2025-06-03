import numpy as np
import os
import math
import threading

import sys


on = "EINC"

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
    home = "/wang/users/dominik/cluster_home/einc/Another_try/2d-dipolar-bose-gas-cl-master/"
    sys.path.append(home)

    from mylib import *

    card={ "3090":"rtx3090", "A100": "a100", "splitted": "1g.10gb"}
    c = card["splitted"]

    Run_command = "nohup srun -p einc -A einc --gres=gpu:"+c+":1 singularity exec --app ido /containers/testing/f27_c21380p5_2023-10-16_2.img ./2D_bosegascl"
    
    #"sbatch jobscript.sh --gres=gpu:"+c+":1 --array=" 

    print("Running on "+on +" with " +c+"\n")

#BoseCL_Simulation_T_1.25_mu_0.10526_g_0.77259_M_0.23178_SIGMA_0_D_3.08221_OMEGAX_0_OMEGAY_0_OMEGAZ_0_components_1_lattice_24_24_1_32_Run1

Len_x=64
m = 0.5
g = 0.1
eps_dd = 0

mu = -13

h_l =  (1/math.sqrt(abs(mu)))
d = h_l
#g = (d * math.sqrt(2*math.pi)) * g_2d
M = 3*g*eps_dd



offset=0
N_Runs = 3

compile(on, M, mu, g, d, 0, Len_x, N_Runs, offset, "Makefile", "main.cu", "Parameters.h", c)
#for runs in np.arange(0, N_Runs, 1):
#    runner = "./2D_bosegascl"+str(runs)
#    os.system(runner)
