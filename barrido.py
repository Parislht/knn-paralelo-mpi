import argparse
import csv
import os
import shutil
import subprocess
import sys
import time

from generar_datos import nombreArchivo

REJILLAS = {1: [(1, 1)], 2: [(1, 2), (2, 1)], 4: [(1, 4), (2, 2), (4, 1)], 8: [(1, 8), (2, 4), (4, 2), (8, 1)]}


def construirConfiguraciones(rapido):
    if rapido:
        return [("rapida", 2000, 0.2, pr, pc) for p in (1, 2) for pr, pc in REJILLAS[p]]
    configuraciones = []
    for n in (2000, 4000, 8000, 16000, 32000):
        for p in (1, 2, 4, 8):
            for pr, pc in REJILLAS[p]:
                configuraciones.append(("principal", n, 0.2, pr, pc))
    for n, p in ((11314, 2), (22627, 8)):
        for pr, pc in REJILLAS[p]:
            configuraciones.append(("debil", n, 0.2, pr, pc))
    for testSize in (0.5, 0.8):
        for pr, pc in REJILLAS[8]:
            configuraciones.append(("razon", 16000, testSize, pr, pc))
    return configuraciones


def configuracionesTerminadas(rutaCsv, reps):
    if not os.path.exists(rutaCsv):
        return set()
    conteo = {}
    with open(rutaCsv, newline="") as archivo:
        for fila in csv.DictReader(archivo):
            clave = (fila["etiqueta"], int(fila["N"]), float(fila["testSize"]), int(fila["pr"]), int(fila["pc"]))
            conteo[clave] = conteo.get(clave, 0) + 1
    return {clave for clave, veces in conteo.items() if veces >= reps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rapido", action="store_true")
    argumentos = parser.parse_args()

    mpiexec = shutil.which("mpiexec")
    if mpiexec is None:
        print("ERROR: no se encontro mpiexec. Activa el entorno y revisa la instalacion de MS-MPI.")
        sys.exit(1)

    reps = 1 if argumentos.rapido else 3
    rutaCsv = "prueba_rapida_v2.csv" if argumentos.rapido else "resultados_v2.csv"
    configuraciones = construirConfiguraciones(argumentos.rapido)
    terminadas = configuracionesTerminadas(rutaCsv, reps)

    entorno = dict(os.environ)
    for variableHilos in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        entorno[variableHilos] = "1"

    inicioTotal = time.time()
    for numero, (etiqueta, n, testSize, pr, pc) in enumerate(configuraciones, start=1):
        if (etiqueta, n, round(testSize, 2), pr, pc) in terminadas:
            print(f"[{numero}/{len(configuraciones)}] ya estaba hecha: {etiqueta} N={n} ts={testSize} {pr}x{pc}")
            continue
        rutaDatos = nombreArchivo(n, testSize)
        if not os.path.exists(rutaDatos):
            print(f"ERROR: falta {rutaDatos}. Ejecuta primero: python generar_datos.py")
            sys.exit(1)
        print(f"[{numero}/{len(configuraciones)}] {etiqueta} N={n} ts={testSize} p={pr * pc} rejilla {pr}x{pc}", flush=True)
        inicio = time.time()
        resultado = subprocess.run(
            [mpiexec, "-n", str(pr * pc), sys.executable, "knn_v3.py", "--datos", rutaDatos,
             "--pr", str(pr), "--pc", str(pc), "--reps", str(reps), "--salida", rutaCsv, "--etiqueta", etiqueta],
            env=entorno, capture_output=True, text=True,
        )
        if resultado.returncode != 0:
            print(resultado.stdout)
            print(resultado.stderr)
            print("ERROR en esta configuracion. Envia una captura de este mensaje.")
            sys.exit(1)
        print(resultado.stdout.rstrip())
        transcurrido = time.time() - inicioTotal
        print(f"    configuracion: {time.time() - inicio:.1f} s | acumulado: {transcurrido / 60:.1f} min", flush=True)

    print(f"\nBARRIDO TERMINADO en {(time.time() - inicioTotal) / 60:.1f} min. Resultados en {rutaCsv}")


if __name__ == "__main__":
    main()
