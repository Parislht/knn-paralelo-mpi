import os
import shutil
import subprocess
import sys

import numpy as np

from knn_comun import cargarDatos
from knn_enunciado import predecirEnunciado

CARPETA = "validacion"
PRUEBAS = [
    ("v1 (DAG, A)", 1, ["knn_v1.py"]),
    ("v1 (DAG, A)", 4, ["knn_v1.py"]),
    ("v1 (DAG, A)", 8, ["knn_v1.py"]),
    ("v2 A", 4, ["knn_v2.py", "--estrategia", "A"]),
    ("v2 B", 4, ["knn_v2.py", "--estrategia", "B"]),
    ("v2 A", 8, ["knn_v2.py", "--estrategia", "A"]),
    ("v2 B", 8, ["knn_v2.py", "--estrategia", "B"]),
    ("v3 1x1", 1, ["knn_v3.py", "--pr", "1", "--pc", "1", "--reps", "1"]),
    ("v3 1x8 (A)", 8, ["knn_v3.py", "--pr", "1", "--pc", "8", "--reps", "1"]),
    ("v3 2x4", 8, ["knn_v3.py", "--pr", "2", "--pc", "4", "--reps", "1"]),
    ("v3 4x2", 8, ["knn_v3.py", "--pr", "4", "--pc", "2", "--reps", "1"]),
    ("v3 8x1 (B)", 8, ["knn_v3.py", "--pr", "8", "--pc", "1", "--reps", "1"]),
]


def main():
    mpiexec = shutil.which("mpiexec")
    if mpiexec is None:
        print("ERROR: no se encontro mpiexec. Activa el entorno y revisa la instalacion de MS-MPI.")
        sys.exit(1)
    os.makedirs(CARPETA, exist_ok=True)

    etiquetasReales = cargarDatos()[3]
    print("Ejecutando el codigo secuencial del enunciado...")
    prediccionesEnunciado, accuracyEnunciado, tiempoEnunciado = predecirEnunciado()
    lineas = [
        f"Codigo del enunciado: accuracy={accuracyEnunciado:.4f}, tiempo={tiempoEnunciado:.2f} s, {len(prediccionesEnunciado)} predicciones",
        "",
        f"{'Version':<14}{'p':>3}  {'Accuracy':>9}  {'Diferencias con el enunciado':>28}",
    ]
    todoCoincide = True
    for numero, (nombre, procesos, comando) in enumerate(PRUEBAS):
        rutaPred = os.path.join(CARPETA, f"pred_{numero}.npy")
        resultado = subprocess.run(
            [mpiexec, "-n", str(procesos), sys.executable] + comando + ["--guardarPred", rutaPred],
            capture_output=True, text=True,
        )
        if resultado.returncode != 0 or not os.path.exists(rutaPred):
            lineas.append(f"{nombre:<14}{procesos:>3}  ERROR: {resultado.stderr.strip()[-300:]}")
            todoCoincide = False
            print(lineas[-1])
            continue
        predicciones = np.load(rutaPred)
        diferencias = int(np.sum(predicciones != prediccionesEnunciado))
        accuracy = float(np.mean(predicciones == etiquetasReales))
        todoCoincide = todoCoincide and diferencias == 0
        lineas.append(f"{nombre:<14}{procesos:>3}  {accuracy:>9.4f}  {diferencias:>28}")
        print(lineas[-1])

    lineas.append("")
    lineas.append("RESULTADO: TODAS LAS VERSIONES COINCIDEN CON EL ENUNCIADO" if todoCoincide else "RESULTADO: HAY DIFERENCIAS (revisar)")
    with open("validacion.txt", "w", encoding="utf-8") as archivo:
        archivo.write("\n".join(lineas) + "\n")
    print("\n".join(lineas[:1] + lineas[-1:]))
    print("Resumen guardado en validacion.txt")


if __name__ == "__main__":
    main()
