#!/bin/bash
#SBATCH --job-name=knn-prueba
#SBATCH --partition=debug
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=8G
#SBATCH --output=salida_prueba_%j.txt

source "$HOME/knn-entorno.sh"
cd "$HOME/knn-khipu"
echo "Nodo: $(hostname)"
ldd $(python -c "import mpi4py,glob,os;print(glob.glob(os.path.join(os.path.dirname(mpi4py.__file__),'MPI*.so'))[0])") | grep -E "ucp|ucs|mpi|not found"
mpiexec -n 4 python verificar_mpi.py
python validar.py
python generar_datos.py
python barrido_khipu.py --modo prueba
