import os
import time

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

CARPETA = "datos"
SEMILLA = 2026
CONFIGURACIONES = [
    (2000, 0.2), (4000, 0.2), (8000, 0.2), (11314, 0.2), (16000, 0.2),
    (22627, 0.2), (32000, 0.2), (16000, 0.5), (16000, 0.8),
]


def nombreArchivo(n, testSize):
    return os.path.join(CARPETA, f"N{n}_ts{int(round(testSize * 100))}.npz")


def aumentar(x, y, cantidad, generador):
    if cantidad <= len(x):
        return x[:cantidad].copy(), y[:cantidad].copy()
    extra = cantidad - len(x)
    base = generador.integers(0, len(x), extra)
    imagenes = x[base].reshape(-1, 8, 8)
    conBorde = np.pad(imagenes, ((0, 0), (1, 1), (1, 1)))
    desplazX = generador.integers(-1, 2, extra)
    desplazY = generador.integers(-1, 2, extra)
    nuevas = np.empty_like(imagenes)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            seleccion = (desplazX == dx) & (desplazY == dy)
            nuevas[seleccion] = conBorde[seleccion, 1 + dy:9 + dy, 1 + dx:9 + dx]
    nuevas = np.clip(nuevas + generador.normal(0.0, 1.0, nuevas.shape), 0.0, 16.0)
    return np.vstack([x, nuevas.reshape(extra, 64)]), np.concatenate([y, y[base]])


def main():
    os.makedirs(CARPETA, exist_ok=True)
    digitos = load_digits()
    inicio = time.time()
    for n, testSize in CONFIGURACIONES:
        xTrOrig, xTeOrig, yTrOrig, yTeOrig = train_test_split(
            digitos.data, digitos.target, test_size=testSize, random_state=42
        )
        generador = np.random.default_rng(SEMILLA + n + int(testSize * 100))
        nTe = int(round(testSize * n))
        nTr = n - nTe
        xTr, yTr = aumentar(xTrOrig, yTrOrig, nTr, generador)
        xTe, yTe = aumentar(xTeOrig, yTeOrig, nTe, generador)
        np.savez(nombreArchivo(n, testSize), xTr=xTr, yTr=yTr, xTe=xTe, yTe=yTe)
        print(f"{nombreArchivo(n, testSize)}: nTr={nTr}, nTe={nTe}")
    print(f"Listo: {len(CONFIGURACIONES)} conjuntos en la carpeta '{CARPETA}' ({time.time() - inicio:.1f} s)")


if __name__ == "__main__":
    main()
