import os

for variableHilos in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variableHilos] = "1"

from collections import Counter

import numpy as np

K = 3
VALORES_POR_LOTE = 250_000


def cargarDatos(ruta=None, testSize=0.2):
    if ruta is None:
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split

        digitos = load_digits()
        xTr, xTe, yTr, yTe = train_test_split(digitos.data, digitos.target, test_size=testSize, random_state=42)
    else:
        archivo = np.load(ruta)
        xTr, xTe, yTr, yTe = archivo["xTr"], archivo["xTe"], archivo["yTr"], archivo["yTe"]
    return (
        np.ascontiguousarray(xTr, dtype=np.float64),
        np.ascontiguousarray(xTe, dtype=np.float64),
        np.asarray(yTr, dtype=np.int64),
        np.asarray(yTe, dtype=np.int64),
    )


def topKLocal(bloqueTe, bloqueTr, etiquetasTr, indicesTr, k=K, valoresPorLote=VALORES_POR_LOTE):
    mTe, mTr = len(bloqueTe), len(bloqueTr)
    kLocal = min(k, mTr)
    distK = np.empty((mTe, kLocal))
    idxK = np.empty((mTe, kLocal), dtype=np.int64)
    labK = np.empty((mTe, kLocal), dtype=np.int64)
    if mTe == 0 or kLocal == 0:
        return distK, idxK, labK
    dimension = bloqueTe.shape[1]
    consultasPorLote = max(1, min(mTe, valoresPorLote // mTr))
    puntosPorTrozo = max(1, valoresPorLote // (consultasPorLote * dimension))
    for inicio in range(0, mTe, consultasPorLote):
        lote = bloqueTe[inicio:inicio + consultasPorLote]
        distancias = np.empty((len(lote), mTr))
        for desde in range(0, mTr, puntosPorTrozo):
            trozo = bloqueTr[desde:desde + puntosPorTrozo]
            diferencia = lote[:, None, :] - trozo[None, :, :]
            distancias[:, desde:desde + len(trozo)] = np.sqrt(np.einsum("ijk,ijk->ij", diferencia, diferencia))
        if kLocal < mTr:
            elegidos = np.argpartition(distancias, kLocal - 1, axis=1)[:, :kLocal]
        else:
            elegidos = np.tile(np.arange(mTr), (len(lote), 1))
        distElegidas = np.take_along_axis(distancias, elegidos, axis=1)
        idxElegidos = indicesTr[elegidos]
        orden = np.lexsort((idxElegidos, distElegidas), axis=1)
        fin = inicio + len(lote)
        distK[inicio:fin] = np.take_along_axis(distElegidas, orden, axis=1)
        idxK[inicio:fin] = np.take_along_axis(idxElegidos, orden, axis=1)
        labK[inicio:fin] = np.take_along_axis(etiquetasTr[elegidos], orden, axis=1)
    return distK, idxK, labK


def votar(etiquetasOrdenadas):
    return np.array([Counter(fila.tolist()).most_common(1)[0][0] for fila in etiquetasOrdenadas], dtype=np.int64)


def fusionarYVotar(listaCandidatos, k=K):
    distTodas = np.concatenate([c[0] for c in listaCandidatos], axis=1)
    idxTodos = np.concatenate([c[1] for c in listaCandidatos], axis=1)
    labTodas = np.concatenate([c[2] for c in listaCandidatos], axis=1)
    orden = np.lexsort((idxTodos, distTodas), axis=1)[:, :k]
    return votar(np.take_along_axis(labTodas, orden, axis=1))
