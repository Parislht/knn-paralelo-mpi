import argparse
from collections import Counter

import numpy as np
from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def distanciaEuclidiana(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--guardarPred", default=None)
    argumentos = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    k = 3

    piezasTr = None
    xTe = None
    yTe = None
    if rank == 0:
        digitos = load_digits()
        xTr, xTe, yTr, yTe = train_test_split(digitos.data, digitos.target, test_size=0.2, random_state=42)
        indicesTr = np.arange(len(xTr))
        piezasTr = list(zip(np.array_split(xTr, size), np.array_split(yTr, size), np.array_split(indicesTr, size)))

    comm.Barrier()
    inicio = MPI.Wtime()

    bloqueTr, bloqueY, bloqueIdx = comm.scatter(piezasTr, root=0)
    xTe = comm.bcast(xTe, root=0)

    candidatosLocales = []
    for puntoPrueba in xTe:
        distancias = [distanciaEuclidiana(puntoPrueba, x) for x in bloqueTr]
        mejores = np.argsort(distancias)[:k]
        candidatosLocales.append([(distancias[i], int(bloqueIdx[i]), int(bloqueY[i])) for i in mejores])

    todosLosCandidatos = comm.gather(candidatosLocales, root=0)

    if rank == 0:
        prediccion = []
        for t in range(len(xTe)):
            unionCandidatos = []
            for candidatosProceso in todosLosCandidatos:
                unionCandidatos.extend(candidatosProceso[t])
            unionCandidatos.sort(key=lambda c: (c[0], c[1]))
            etiquetasVecinos = [c[2] for c in unionCandidatos[:k]]
            prediccion.append(Counter(etiquetasVecinos).most_common(1)[0][0])
        prediccion = np.array(prediccion)
        fin = MPI.Wtime()
        accuracy = np.mean(prediccion == yTe)
        print(f"v1 (DAG, estrategia A) p={size} | accuracy={accuracy:.4f} | tiempo={fin - inicio:.4f} s")
        if argumentos.guardarPred:
            np.save(argumentos.guardarPred, prediccion)


if __name__ == "__main__":
    main()
