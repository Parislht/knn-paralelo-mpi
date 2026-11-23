import platform
import sys

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
errores = []

datoRaiz = {"mensaje": "hola", "valor": 42} if rank == 0 else None
datoRecibido = comm.bcast(datoRaiz, root=0)
if datoRecibido != {"mensaje": "hola", "valor": 42}:
    errores.append("bcast")

bloques = np.array_split(np.arange(1000), size) if rank == 0 else None
miBloque = comm.scatter(bloques, root=0)
sumasParciales = comm.gather(int(miBloque.sum()), root=0)
if rank == 0 and sum(sumasParciales) != sum(range(1000)):
    errores.append("scatter/gather")

arregloNumpy = np.arange(10, dtype="float64") if rank == 0 else np.empty(10, dtype="float64")
comm.Bcast(arregloNumpy, root=0)
if not np.array_equal(arregloNumpy, np.arange(10, dtype="float64")):
    errores.append("Bcast con NumPy")

procesosPorFila = size // 2 if size % 2 == 0 else size
filasRejilla = size // procesosPorFila
miFila = rank // procesosPorFila
miColumna = rank % procesosPorFila
filaComm = comm.Split(miFila, rank)
colComm = comm.Split(miColumna, rank)
if filaComm.Get_size() != procesosPorFila or colComm.Get_size() != filasRejilla:
    errores.append("Split (tamanos)")
sumaFila = filaComm.allreduce(rank, op=MPI.SUM)
esperado = sum(miFila * procesosPorFila + j for j in range(procesosPorFila))
if sumaFila != esperado:
    errores.append("Split (contenido)")

try:
    from sklearn.datasets import load_digits
    formaDatos = load_digits().data.shape
    if formaDatos != (1797, 64):
        errores.append("load_digits forma " + str(formaDatos))
except Exception as excepcion:
    errores.append("scikit-learn: " + str(excepcion))

inicio = MPI.Wtime()
comm.Barrier()
if MPI.Wtime() - inicio < 0:
    errores.append("Wtime")

todosLosErrores = comm.gather(errores, root=0)
nombresNodo = comm.gather(MPI.Get_processor_name(), root=0)

if rank == 0:
    import sklearn
    import mpi4py
    print("=" * 60)
    print("Biblioteca MPI :", MPI.Get_library_version().splitlines()[0].strip())
    print("Python         :", sys.version.split()[0], platform.architecture()[0])
    print("mpi4py         :", mpi4py.__version__)
    print("NumPy          :", np.__version__)
    print("scikit-learn   :", sklearn.__version__)
    print("Procesos (p)   :", size, "| rejilla de prueba", filasRejilla, "x", procesosPorFila)
    print("Equipo         :", sorted(set(nombresNodo)))
    print("=" * 60)
    hayErrores = False
    for r, listaErrores in enumerate(todosLosErrores):
        if listaErrores:
            hayErrores = True
            print("Proceso", r, "FALLO en:", ", ".join(listaErrores))
    if hayErrores:
        print("RESULTADO: HAY ERRORES (ver arriba)")
    else:
        print("RESULTADO: TODO OK -", size, "procesos comunicandose correctamente")
