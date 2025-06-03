import os
import random

mu_values = [-2, -1.5]
P_values = [0]
N_runs = 3

for mu in mu_values:
    for P in P_values:
        for run in range(N_runs):
            seed = str(int(random.random()*1e16))
            seed2 = str(int(random.random()*1e16))
            with open("params.txt", "w") as f:
                f.write(f"mu={mu}\nP={P}\nrun={run}\nseed={seed}\nseed2={seed2}\n")

            # Wähle die passende Executable
            cmd = f"./2D_bosegascl{run}"
            print(f"Running: {cmd} for mu={mu}, P={P}, run={run}")
            os.system(cmd)