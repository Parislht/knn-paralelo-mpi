#!/bin/bash
#SBATCH --job-name=knn-2nodos
#SBATCH --partition=standard
#SBATCH --time=00:45:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=24G
#SBATCH --output=salida_dosnodos_%j.txt

source "$HOME/knn-entorno.sh"
cd "$HOME/knn-khipu"
echo "Nodos: $SLURM_JOB_NODELIST"
mpiexec -n 4 -ppn 2 python -c "from mpi4py import MPI; print('rank', MPI.COMM_WORLD.Get_rank(), 'en', MPI.Get_processor_name())"
python barrido_khipu.py --modo dosnodos
