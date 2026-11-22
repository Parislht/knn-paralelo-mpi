import argparse
import csv
import os

from knn_comun import K, cargarDatos, fusionarYVotar, topKLocal

import numpy as np
from mpi4py import MPI

COLUMNAS = [
    "etiqueta", "N", "nTr", "nTe", "testSize", "p", "pr", "pc", "rep",
    "tScatterTr", "tBcastTr", "tScatterTe", "tBcastTe", "tComputo",
    "tGatherCand", "tFusion", "tGatherPred", "tComm", "tComp", "tTotal", "accuracy",
]


def ejecutarUnaVez(comm, filaComm, colComm, i, j, pr, pc, datosRaiz):
    tiempos = {}

    def marcar(nombre, inicio):
        comm.Barrier()
        tiempos[nombre] = MPI.Wtime() - inicio
        return MPI.Wtime()

    comm.Barrier()
    reloj = MPI.Wtime()

    bloqueTr = None
    if i == 0:
        piezasTr = None
        if filaComm.Get_rank() == 0:
            xTr, xTe, yTr, yTe = datosRaiz
            indices = np.arange(len(xTr))
            piezasTr = list(zip(np.array_split(xTr, pc), np.array_split(yTr, pc), np.array_split(indices, pc)))
        bloqueTr = filaComm.scatter(piezasTr, root=0)
    reloj = marcar("tScatterTr", reloj)
    bloqueTr = colComm.bcast(bloqueTr, root=0)
    reloj = marcar("tBcastTr", reloj)

    bloqueTe = None
    if j == 0:
        piezasTe = None
        if colComm.Get_rank() == 0:
            piezasTe = np.array_split(datosRaiz[1], pr)
        bloqueTe = colComm.scatter(piezasTe, root=0)
    reloj = marcar("tScatterTe", reloj)
    bloqueTe = filaComm.bcast(bloqueTe, root=0)
    reloj = marcar("tBcastTe", reloj)

    misTr, misY, misIdx = bloqueTr
    candidatos = topKLocal(bloqueTe, misTr, misY, misIdx, K)
    reloj = marcar("tComputo", reloj)

    candidatosFila = filaComm.gather(candidatos, root=0)
    reloj = marcar("tGatherCand", reloj)

    prediccionFila = fusionarYVotar(candidatosFila, K) if j == 0 else None
    reloj = marcar("tFusion", reloj)

    prediccion = None
    if j == 0:
        partes = colComm.gather(prediccionFila, root=0)
        if comm.Get_rank() == 0:
            prediccion = np.concatenate(partes)
    marcar("tGatherPred", reloj)
    return tiempos, prediccion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datos", default=None)
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--pc", type=int, required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--calentamiento", type=int, default=1)
    parser.add_argument("--salida", default=None)
    parser.add_argument("--etiqueta", default="")
    parser.add_argument("--guardarPred", default=None)
    argumentos = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    pr, pc = argumentos.pr, argumentos.pc
    if pr * pc != size:
        if rank == 0:
            print(f"ERROR: pr*pc = {pr * pc} pero se lanzaron {size} procesos")
        comm.Abort(1)

    i, j = rank // pc, rank % pc
    filaComm = comm.Split(i, rank)
    colComm = comm.Split(j, rank)

    datosRaiz = cargarDatos(argumentos.datos) if rank == 0 else None

    for repeticion in range(argumentos.calentamiento + argumentos.reps):
        tiempos, prediccion = ejecutarUnaVez(comm, filaComm, colComm, i, j, pr, pc, datosRaiz)
        if rank != 0 or repeticion < argumentos.calentamiento:
            continue
        xTr, xTe, yTr, yTe = datosRaiz
        accuracy = float(np.mean(prediccion == yTe))
        tiempoComm = sum(tiempos[c] for c in ("tScatterTr", "tBcastTr", "tScatterTe", "tBcastTe", "tGatherCand", "tGatherPred"))
        tiempoComp = tiempos["tComputo"] + tiempos["tFusion"]
        nTr, nTe = len(xTr), len(xTe)
        fila = {
            "etiqueta": argumentos.etiqueta, "N": nTr + nTe, "nTr": nTr, "nTe": nTe,
            "testSize": round(nTe / (nTr + nTe), 2), "p": size, "pr": pr, "pc": pc,
            "rep": repeticion - argumentos.calentamiento + 1,
            **{c: f"{tiempos[c]:.6f}" for c in tiempos},
            "tComm": f"{tiempoComm:.6f}", "tComp": f"{tiempoComp:.6f}",
            "tTotal": f"{tiempoComm + tiempoComp:.6f}", "accuracy": f"{accuracy:.4f}",
        }
        print(
            f"N={nTr + nTe} p={size} rejilla {pr}x{pc} rep {fila['rep']} | "
            f"Tcomp={tiempoComp:.3f} s Tcomm={tiempoComm:.3f} s Ttotal={tiempoComm + tiempoComp:.3f} s | acc={accuracy:.4f}",
            flush=True,
        )
        if argumentos.salida:
            existe = os.path.exists(argumentos.salida)
            with open(argumentos.salida, "a", newline="") as archivo:
                escritor = csv.DictWriter(archivo, fieldnames=COLUMNAS)
                if not existe:
                    escritor.writeheader()
                escritor.writerow(fila)
        if argumentos.guardarPred and repeticion == argumentos.calentamiento:
            np.save(argumentos.guardarPred, prediccion)


if __name__ == "__main__":
    main()
