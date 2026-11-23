module load mpich/3.4.3-ucx
module load python3/3.11.11
export LD_LIBRARY_PATH=$HOME/knn-ucx:$LD_LIBRARY_PATH
export UCX_MODULE_DIR=$HOME/knn-ucx/ucx
source "$HOME/knn-env/bin/activate"
