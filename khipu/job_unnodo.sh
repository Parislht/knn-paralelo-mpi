#!/bin/bash
#SBATCH --job-name=knn-1nodo
#SBATCH --partition=standard
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=48G
#SBATCH --output=salida_unnodo_%j.txt

source "$HOME/knn-entorno.sh"
cd "$HOME/knn-khipu"
echo "Nodo: $(hostname)"
lscpu > hardware_unnodo.txt
python barrido_khipu.py --modo unnodo
