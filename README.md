# Pruebas en el clúster Khipu (UTEC)

Esta carpeta tiene los scripts y resultados de las pruebas en Khipu (sección "Ejecución en uno y dos nodos" del informe). El código del KNN es el mismo de la raíz del repositorio (`knn_v3.py`, `knn_comun.py`, etc.), sin cambios.

## Contenido

| Archivo | Qué es |
|---|---|
| `barrido_khipu.py` | Recorre las configuraciones (N, p, rejilla) llamando a `mpiexec ... knn_v3.py` y guarda un CSV; si se corta, continúa donde se quedó |
| `job_prueba.sh` | Trabajo de Slurm: verifica MPI, valida contra el enunciado, genera los datos y hace una prueba corta |
| `job_unnodo.sh` | Barrido en un nodo: N = 16 000 y 32 000, p = 1 a 16 |
| `job_dosnodos.sh` | Barrido en dos nodos: N = 32 000, p = 8 y 16, mitad de procesos en cada nodo |
| `entorno.sh` | Carga los módulos y activa el entorno virtual dentro de cada trabajo |
| `verificar_mpi.py` | Prueba mínima de comunicación entre procesos |
| `graficas_khipu.py` | Genera las figuras de Khipu del informe |
| `resultados_khipu_*.csv` | Resultados medidos (3 repeticiones por configuración) |
| `hardware_unnodo.txt` | Salida de `lscpu` del nodo de cómputo |

## Cómo se ejecutó

1. Copiar a una carpeta `~/knn-khipu` del clúster los archivos de la raíz del repositorio junto con los de esta carpeta, y `entorno.sh` a `~/knn-entorno.sh`.
2. En el nodo de acceso, crear el entorno:
   ```bash
   module load python3/3.11.11
   python3 -m venv ~/knn-env && source ~/knn-env/bin/activate
   pip install numpy scikit-learn pandas matplotlib
   MPICC=mpicc pip install --no-binary=mpi4py mpi4py
   ```
3. Las librerías de UCX que usa MPICH solo están en el nodo de acceso, así que se copian a la home (que es compartida con los nodos de cómputo):
   ```bash
   mkdir -p ~/knn-ucx
   cp -L /lib64/libucp.so.0 /lib64/libucs.so.0 /lib64/libuct.so.0 /lib64/libucm.so.0 ~/knn-ucx/
   ```
4. Enviar los trabajos: `sbatch job_prueba.sh`, luego `sbatch job_unnodo.sh` y `sbatch job_dosnodos.sh`.

La cuenta de estudiante permite 32 CPUs lógicas y cada núcleo físico cuenta como dos (hyperthreading). Por eso se usan como máximo 16 procesos por nodo con `--hint=nomultithread`, uno por núcleo físico.

Para las figuras, desde esta carpeta: `python graficas_khipu.py` (usa `../resultados_v2.csv` para comparar con el equipo local).
