#!/bin/bash -exu

########################################
######### SLURM-Header #################
########################################

#SBATCH --partition=einc
#SBATCH --account=einc
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --mem=32gb
#SBATCH --job-name=CL_1D
#SBATCH --output=out_%j.log
#SBATCH --error=err_%j.log
#SBATCH --time=24:00:00

echo "Working Directory: $PWD"
echo "Job id:           $SLURM_JOB_ID"
echo "Job name:         $SLURM_JOB_NAME"
echo "Nodes/GPUs:       $SLURM_JOB_NUM_NODES  /  $SLURM_JOB_GPUS"

########################################
######### Umgebung setzen ##############
########################################

export WORKSPACE=$PWD
cmdwrap="apptainer exec --nv --app ido /containers/testing/f27_c21380p5_2023-10-16_2.img"

# Falls du besondere Env- oder LIB-Pfade brauchst, wie in deinem Beispiel:
export LD_LIBRARY_PATH="/opt/...mkl/lib/intel64_lin:$LD_LIBRARY_PATH"
export C_INCLUDE_PATH="/opt/...mkl/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="/opt/...mkl/include:${CPLUS_INCLUDE_PATH:-}"
export OMP_NUM_THREADS=4

########################################
######### Kompilieren ##################
########################################

echo ">>> compile for all N_runs"
# Kompilierskript läuft innerhalb des Containers
time $cmdwrap bash -c "cd $WORKSPACE && python3 compile_script.py"

########################################
######### Parameter-Sweep ##############
########################################

echo ">>> start reading new parameter"
time $cmdwrap bash -c "cd $WORKSPACE && python3 run_all.py"

echo ">>> Job finished."