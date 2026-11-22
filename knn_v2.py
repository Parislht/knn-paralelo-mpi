import argparse

from knn_comun import K, cargarDatos, fusionarYVotar, topKLocal, votar

import numpy as np
from mpi4py import MPI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estrategia", choices=["A", "B"], required=True)
    parser.add_argument("--datos", default=None)
    parser.add_argument("--guardarPred", default=None)
    argumentos = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    datos = cargarDatos(argumentos.datos) if rank == 0 else None
    tiempos = {}

    def marcar(nombre, inicio):
        comm.Barrier()
        tiempos[nombre] = MPI.Wtime() - inicio
        return MPI.Wtime()

    comm.Barrier()
    reloj = MPI.Wtime()

    if argumentos.estrategia == "A":
        piezas = None
        xTe = None
        if rank == 0:
            xTr, xTe, yTr, yTe = datos
            indices = np.arange(len(xTr))
            piezas = list(zip(np.array_split(xTr, size), np.array_split(yTr, size), np.array_split(indices, size)))
        bloqueTr, bloqueY, bloqueIdx = comm.scatter(piezas, root=0)
        reloj = marcar("scatter", reloj)
        xTe = comm.bcast(xTe, root=0)
        reloj = marcar("bcast", reloj)
        candidatos = topKLocal(xTe, bloqueTr, bloqueY, bloqueIdx, K)
        reloj = marcar("computo", reloj)
        todos = comm.gather(candidatos, root=0)
        reloj = marcar("gather", reloj)
        prediccion = fusionarYVotar(todos, K) if rank == 0 else None
        reloj = marcar("fusion", reloj)
    else:
        piezas = None
        entrenamiento = None
        if rank == 0:
            xTr, xTe, yTr, yTe = datos
            piezas = np.array_split(xTe, size)
            entrenamiento = (xTr, yTr, np.arange(len(xTr)))
        bloqueTe = comm.scatter(piezas, root=0)
        reloj = marcar("scatter", reloj)
        xTrCompleto, yTrCompleto, idxCompleto = comm.bcast(entrenamiento, root=0)
        reloj = marcar("bcast", reloj)
        distK, idxK, labK = topKLocal(bloqueTe, xTrCompleto, yTrCompleto, idxCompleto, K)
        reloj = marcar("computo", reloj)
        prediccionLocal = votar(labK)
        reloj = marcar("fusion", reloj)
        partes = comm.gather(prediccionLocal, root=0)
        reloj = marcar("gather", reloj)
        prediccion = np.concatenate(partes) if rank == 0 else None

    if rank == 0:
        yTe = datos[3]
        accuracy = np.mean(prediccion == yTe)
        tiempoComm = tiempos["scatter"] + tiempos["bcast"] + tiempos["gather"]
        tiempoComp = tiempos["computo"] + tiempos["fusion"]
        print(
            f"v2 estrategia {argumentos.estrategia} p={size} | accuracy={accuracy:.4f} | "
            f"Tcomp={tiempoComp:.4f} s | Tcomm={tiempoComm:.4f} s | Ttotal={tiempoComp + tiempoComm:.4f} s"
        )
        if argumentos.guardarPred:
            np.save(argumentos.guardarPred, prediccion)


if __name__ == "__main__":
    main()
