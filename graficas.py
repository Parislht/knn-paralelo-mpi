import math
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

AZUL, NARANJA, AQUA, AMARILLO, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
CARPETA = "figuras"
DIMENSION, K = 64, 3

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": "#b9b8b3",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6, "lines.linewidth": 2, "legend.frameon": False,
})


def techoLog2(q):
    return math.ceil(math.log2(q)) if q > 1 else 0


def cargarMedianas(ruta):
    datos = pd.read_csv(ruta)
    claves = ["etiqueta", "N", "nTr", "nTe", "testSize", "p", "pr", "pc"]
    medianas = datos.groupby(claves).median(numeric_only=True).reset_index()
    medianas["F"] = 3.0 * DIMENSION * medianas.nTe * medianas.nTr
    return medianas


def ejeProcesos(eje):
    eje.set_xscale("log", base=2)
    eje.set_xticks([1, 2, 4, 8])
    eje.set_xticklabels(["1", "2", "4", "8"])
    eje.set_xlabel("Procesos $p$")


def ajustarConstantes(medianas, principal):
    secuencial = principal[principal.p == 1]
    tiempoPorFlop = float((secuencial.F * secuencial.tTotal).sum() / (secuencial.F ** 2).sum())
    paralelas = medianas[medianas.p > 1].copy()
    paralelas["latencias"] = paralelas.apply(
        lambda f: 2 * (f.pr - 1) + 2 * (f.pc - 1) + techoLog2(f.pr) + techoLog2(f.pc), axis=1)
    paralelas["volumen"] = paralelas.apply(
        lambda f: ((f.pc - 1) / f.pc) * f.nTr * DIMENSION + ((f.pr - 1) / f.pr) * f.nTe * DIMENSION
        + techoLog2(f.pr) * f.nTr * DIMENSION / f.pc + techoLog2(f.pc) * f.nTe * DIMENSION / f.pr
        + (f.pc - 1) * K * f.nTe / f.pr + ((f.pr - 1) / f.pr) * f.nTe, axis=1)
    matriz = np.c_[paralelas.latencias, paralelas.volumen]
    (latencia, costoPorValor), *_ = np.linalg.lstsq(matriz, paralelas.tComm, rcond=None)
    prediccion = matriz @ [latencia, costoPorValor]
    r2 = 1 - ((paralelas.tComm - prediccion) ** 2).sum() / ((paralelas.tComm - paralelas.tComm.mean()) ** 2).sum()
    candidatos = medianas.pc * K * medianas.nTe / medianas.pr
    tiempoFusion = float((candidatos * medianas.tFusion).sum() / (candidatos ** 2).sum())
    return tiempoPorFlop, latencia, costoPorValor, r2, tiempoFusion


def figuraSecuencial(principal, tiempoPorFlop):
    secuencial = principal[principal.p == 1].sort_values("N")
    figura, eje = plt.subplots(figsize=(4.6, 2.9))
    xs = np.logspace(np.log10(secuencial.F.min() * 0.7), np.log10(secuencial.F.max() * 1.4), 50)
    eje.plot(xs / 1e9, tiempoPorFlop * xs, color=GRIS, lw=1.5, ls="--",
             label=f"Ajuste $T_s=t_c\\,F$, $t_c={tiempoPorFlop * 1e9:.3f}$ ns")
    eje.plot(secuencial.F / 1e9, secuencial.tTotal, "o", color=AZUL, ms=7, mec="#fcfcfb", mew=1.5, label="Medido ($p=1$)")
    for _, fila in secuencial.iterrows():
        eje.annotate(f"N={int(fila.N / 1000)}k", (fila.F / 1e9, fila.tTotal), textcoords="offset points",
                     xytext=(6, -10), fontsize=7.5, color="#52514e")
    eje.set_xscale("log")
    eje.set_yscale("log")
    eje.set_xlabel("FLOPs de la región paralelizable, $F$ ($\\times 10^9$)")
    eje.set_ylabel("$T_s$ (s)")
    eje.legend(loc="upper left", fontsize=8)
    figura.tight_layout()
    figura.savefig(os.path.join(CARPETA, "fig_ts.pdf"))
    plt.close(figura)


def figuraComunicacion(estrategiaA):
    figura, eje = plt.subplots(figsize=(4.6, 2.9))
    for n, color, marcador in ((2000, AMARILLO, "D"), (8000, AZUL, "o"), (32000, AQUA, "^")):
        serie = estrategiaA[estrategiaA.N == n].sort_values("p")
        eje.plot(serie.p, 100 * serie.tComm / serie.tTotal, marker=marcador, color=color, ms=6,
                 mec="#fcfcfb", mew=1.2, label=f"N = {n:,}".replace(",", " "))
    ejeProcesos(eje)
    eje.set_ylabel("Comunicación (% de $T_p$)")
    eje.legend(fontsize=8)
    figura.tight_layout()
    figura.savefig(os.path.join(CARPETA, "fig_comm.pdf"))
    plt.close(figura)


def figuraSpeedup(estrategiaA, tiemposSecuenciales):
    figura, (ejeS, ejeE) = plt.subplots(1, 2, figsize=(6.6, 2.8))
    ejeS.plot([1, 2, 4, 8], [1, 2, 4, 8], color=GRIS, ls="--", lw=1.5, label="Ideal")
    ejeE.plot([1, 2, 4, 8], [1, 1, 1, 1], color=GRIS, ls="--", lw=1.5, label="Ideal")
    for n, color, marcador in ((8000, AZUL, "o"), (16000, NARANJA, "s"), (32000, AQUA, "^")):
        serie = estrategiaA[estrategiaA.N == n].sort_values("p")
        speedup = tiemposSecuenciales[n] / serie.tTotal
        etiqueta = f"N = {n:,}".replace(",", " ")
        ejeS.plot(serie.p, speedup, marker=marcador, color=color, ms=6, mec="#fcfcfb", mew=1.2, label=etiqueta)
        ejeE.plot(serie.p, speedup / serie.p, marker=marcador, color=color, ms=6, mec="#fcfcfb", mew=1.2, label=etiqueta)
    for eje, titulo in ((ejeS, "Speedup $S$"), (ejeE, "Eficiencia $E$")):
        ejeProcesos(eje)
        eje.set_ylabel(titulo)
    ejeS.set_ylim(0, 8.5)
    ejeE.set_ylim(0, 1.1)
    ejeS.legend(fontsize=7.5, loc="upper left")
    figura.tight_layout()
    figura.savefig(os.path.join(CARPETA, "fig_speedup.pdf"))
    plt.close(figura)


def figuraFlops(estrategiaA, tiempoPorFlop):
    serie = estrategiaA[estrategiaA.N == 32000].sort_values("p")
    figura, eje = plt.subplots(figsize=(4.6, 2.9))
    eje.plot(serie.p, serie.p / tiempoPorFlop / 1e9, color=GRIS, ls="--", lw=1.5, label="Ideal $p/t_c$")
    velocidad = serie.F / serie.tTotal / 1e9
    eje.plot(serie.p, velocidad, marker="o", color=AZUL, ms=6, mec="#fcfcfb", mew=1.2, label="Medido, $F/T_p$")
    for p, valor in zip(serie.p, velocidad):
        eje.annotate(f"{valor:.2f}", (p, valor), textcoords="offset points", xytext=(4, -13), fontsize=7.5, color="#52514e")
    ejeProcesos(eje)
    eje.set_ylabel("GFLOP/s")
    eje.set_ylim(0, 13.5)
    eje.legend(fontsize=8, loc="upper left")
    figura.tight_layout()
    figura.savefig(os.path.join(CARPETA, "fig_flops.pdf"))
    plt.close(figura)


def figuraRejillas(medianas):
    seleccion = medianas[(medianas.N == 16000) & (medianas.p == 8)]
    rejillas = [(1, 8), (2, 4), (4, 2), (8, 1)]
    figura, eje = plt.subplots(figsize=(5.4, 2.9))
    ancho = 0.26
    for posicion, (proporcion, color) in enumerate(((0.2, AZUL), (0.5, NARANJA), (0.8, AQUA))):
        valores = [1e3 * seleccion[(seleccion.testSize == proporcion) & (seleccion.pr == a) & (seleccion.pc == b)].tComm.iloc[0]
                   for a, b in rejillas]
        eje.bar(np.arange(4) + (posicion - 1) * (ancho + 0.02), valores, width=ancho, color=color,
                label=f"test_size = {proporcion}", edgecolor="#fcfcfb", linewidth=1)
    eje.set_xticks(np.arange(4))
    eje.set_xticklabels(["1×8 (A)", "2×4", "4×2", "8×1 (B)"])
    eje.set_ylabel("$T_{comm}$ (ms)")
    eje.set_xlabel("Rejilla de procesos")
    eje.grid(axis="x", visible=False)
    eje.legend(fontsize=8)
    figura.tight_layout()
    figura.savefig(os.path.join(CARPETA, "fig_rejillas.pdf"))
    plt.close(figura)


def main():
    ruta = sys.argv[1] if len(sys.argv) > 1 else "resultados_v2.csv"
    os.makedirs(CARPETA, exist_ok=True)
    medianas = cargarMedianas(ruta)
    principal = medianas[medianas.etiqueta == "principal"]
    estrategiaA = principal[principal.pr == 1]
    tiemposSecuenciales = principal[principal.p == 1].set_index("N").tTotal

    tiempoPorFlop, latencia, costoPorValor, r2, tiempoFusion = ajustarConstantes(medianas, principal)
    print(f"t_c = {tiempoPorFlop * 1e9:.3f} ns/FLOP ({1 / tiempoPorFlop / 1e9:.2f} GFLOP/s)")
    print(f"a = {latencia * 1e3:.3f} ms, b = {costoPorValor * 1e9:.2f} ns/valor, R2 = {r2:.3f}, t_m = {tiempoFusion * 1e6:.3f} us")

    tabla = principal.copy()
    tabla["S"] = tabla.N.map(tiemposSecuenciales) / tabla.tTotal
    tabla["E"] = tabla.S / tabla.p
    tabla["pctComm"] = 100 * tabla.tComm / tabla.tTotal
    tabla["tcEfectivo"] = tabla.tComp * tabla.p / tabla.F * 1e9
    print(tabla[["N", "p", "pr", "pc", "tComp", "tComm", "tTotal", "S", "E", "pctComm", "tcEfectivo"]].round(3).to_string(index=False))

    figuraSecuencial(principal, tiempoPorFlop)
    figuraComunicacion(estrategiaA)
    figuraSpeedup(estrategiaA, tiemposSecuenciales)
    figuraFlops(estrategiaA, tiempoPorFlop)
    figuraRejillas(medianas)
    print(f"Figuras guardadas en la carpeta '{CARPETA}'")


if __name__ == "__main__":
    main()
