import argparse
import csv
import os
import shutil
import subprocess
import sys
import time

from generar_datos import nombreArchivo

REJILLAS = {
    1: [(1, 1)],
    2: [(1, 2)],
    4: [(1, 4)],
    8: [(1, 8), (2, 4), (8, 1)],
    16: [(1, 16), (2, 8), (4, 4), (16, 1)],
    32: [(1, 32), (2, 16), (4, 8), (32, 1)],
}


def construirConfiguraciones(modo):
    if modo == "prueba":
        return [("prueba", 2000, pr, pc) for p in (1, 2) for pr, pc in REJILLAS[p]]
    if modo == "unnodo":
        return [("unnodo", n, pr, pc) for n in (16000, 32000) for p in (1, 2, 4, 8, 16) for pr, pc in REJILLAS[p]]
    return [("dosnodos", 32000, pr, pc) for p in (8, 16) for pr, pc in REJILLAS[p]]


def configuracionesTerminadas(rutaCsv, reps):
    if not os.path.exists(rutaCsv):
        return set()
    conteo = {}
    with open(rutaCsv, newline="") as archivo:
        for fila in csv.DictReader(archivo):
            clave = (fila["etiqueta"], int(fila["N"]), int(fila["pr"]), int(fila["pc"]))
            conteo[clave] = conteo.get(clave, 0) + 1
    return {clave for clave, veces in conteo.items() if veces >= reps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modo", choices=["prueba", "unnodo", "dosnodos"], required=True)
    argumentos = parser.parse_args()

    mpiexec = shutil.which("mpiexec")
    if mpiexec is None:
        print("ERROR: no se encontro mpiexec. Revisa que el modulo de MPICH este cargado.")
        sys.exit(1)

    reps = 1 if argumentos.modo == "prueba" else 3
    rutaCsv = f"resultados_khipu_{argumentos.modo}.csv"
    configuraciones = construirConfiguraciones(argumentos.modo)
    terminadas = configuracionesTerminadas(rutaCsv, reps)
    nodos = 2 if argumentos.modo == "dosnodos" else 1

    entorno = dict(os.environ)
    for variableHilos in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        entorno[variableHilos] = "1"

    inicioTotal = time.time()
    for numero, (etiqueta, n, pr, pc) in enumerate(configuraciones, start=1):
        if (etiqueta, n, pr, pc) in terminadas:
            print(f"[{numero}/{len(configuraciones)}] ya estaba hecha: N={n} {pr}x{pc}", flush=True)
            continue
        rutaDatos = nombreArchivo(n, 0.2)
        if not os.path.exists(rutaDatos):
            print(f"ERROR: falta {rutaDatos}. Ejecuta primero: python generar_datos.py")
            sys.exit(1)
        procesos = pr * pc
        comando = [mpiexec, "-n", str(procesos)]
        if nodos > 1:
            comando += ["-ppn", str(procesos // nodos)]
        comando += [sys.executable, "knn_v3.py", "--datos", rutaDatos, "--pr", str(pr), "--pc", str(pc),
                    "--reps", str(reps), "--salida", rutaCsv, "--etiqueta", etiqueta]
        print(f"[{numero}/{len(configuraciones)}] {etiqueta} N={n} p={procesos} rejilla {pr}x{pc}", flush=True)
        inicio = time.time()
        resultado = subprocess.run(comando, env=entorno, capture_output=True, text=True)
        if resultado.returncode != 0:
            print(resultado.stdout)
            print(resultado.stderr)
            print("ERROR en esta configuracion.")
            sys.exit(1)
        print(resultado.stdout.rstrip())
        print(f"    configuracion: {time.time() - inicio:.1f} s | acumulado: {(time.time() - inicioTotal) / 60:.1f} min", flush=True)

    print(f"\nBARRIDO TERMINADO en {(time.time() - inicioTotal) / 60:.1f} min. Resultados en {rutaCsv}", flush=True)


if __name__ == "__main__":
    main()
