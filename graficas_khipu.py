import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

AZUL, NARANJA, AQUA, AMARILLO, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
CARPETA = "figuras"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": "#b9b8b3",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6, "lines.linewidth": 2, "legend.frameon": False,
})


def medianas(ruta, etiqueta=None):
    datos = pd.read_csv(ruta)
    if etiqueta:
        datos = datos[datos.etiqueta == etiqueta]
    return datos.groupby(["N", "p", "pr", "pc"]).median(numeric_only=True).reset_index()


def figuraSpeedupKhipu(local, unNodo):
    figura, (ejeS, ejeE) = plt.subplots(1, 2, figsize=(6.6, 2.8))
    procesos = [1, 2, 4, 8, 16]
    ejeS.plot(procesos, procesos, color=GRIS, ls="--", lw=1.5, label="Ideal")
    ejeE.plot(procesos, [1] * 5, color=GRIS, ls="--", lw=1.5, label="Ideal")
    series = (
        (local, 32000, AMARILLO, "D", "Local, N = 32 000"),
        (unNodo, 16000, NARANJA, "s", "Khipu, N = 16 000"),
        (unNodo, 32000, AQUA, "^", "Khipu, N = 32 000"),
    )
    for tabla, n, color, marcador, etiqueta in series:
        serie = tabla[(tabla.N == n) & (tabla.pr == 1)].sort_values("p")
        speedup = serie.tTotal.iloc[0] / serie.tTotal
        ejeS.plot(serie.p, speedup, marker=marcador, color=color, ms=6, mec="#fcfcfb", mew=1.2, label=etiqueta)
        ejeE.plot(serie.p, speedup / serie.p, marker=marcador, color=color, ms=6, mec="#fcfcfb", mew=1.2, label=etiqueta)
    for eje, titulo in ((ejeS, "Speedup $S$"), (ejeE, "Eficiencia $E$")):
        eje.set_xscale("log", base=2)
        eje.set_xticks(procesos)
        eje.set_xticklabels([str(p) for p in procesos])
        eje.set_xlabel("Procesos $p$")
        eje.set_ylabel(titulo)
    ejeS.set_ylim(0, 16.5)
    ejeE.set_ylim(0, 1.1)
    ejeS.legend(fontsize=7.5, loc="upper left")
    figura.tight_layout()
    figura.savefig(os.path.join(CARPETA, "fig_khipu_speedup.pdf"))
    plt.close(figura)


def figuraNodos(unNodo, dosNodos):
    rejillas = [(8, 1, 8), (8, 2, 4), (8, 8, 1), (16, 1, 16), (16, 2, 8), (16, 4, 4), (16, 16, 1)]
    etiquetas = [f"{pr}$\\times${pc}" for _, pr, pc in rejillas]
    def tiempos(tabla):
        filas = tabla[tabla.N == 32000].set_index(["p", "pr", "pc"])
        return [filas.loc[clave, "tTotal"] for clave in rejillas]
    posiciones = np.arange(len(rejillas))
    ancho = 0.38
    figura, eje = plt.subplots(figsize=(6.2, 2.7))
    eje.bar(posiciones - ancho / 2 - 0.01, tiempos(unNodo), width=ancho, color=AZUL, edgecolor="#fcfcfb",
            linewidth=1, label="1 nodo")
    eje.bar(posiciones + ancho / 2 + 0.01, tiempos(dosNodos), width=ancho, color=AQUA, edgecolor="#fcfcfb",
            linewidth=1, label="2 nodos")
    eje.axvline(2.5, color="#b9b8b3", lw=1)
    eje.text(1, 3.35, "$p=8$", ha="center", fontsize=8, color="#52514e")
    eje.text(4.5, 3.35, "$p=16$", ha="center", fontsize=8, color="#52514e")
    eje.set_xticks(posiciones)
    eje.set_xticklabels(etiquetas)
    eje.set_xlabel("Rejilla $p_r\\times p_c$")
    eje.set_ylabel("$T_p$ (s)")
    eje.set_ylim(0, 3.6)
    eje.grid(axis="x", visible=False)
    eje.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    figura.tight_layout()
    figura.savefig(os.path.join(CARPETA, "fig_khipu_nodos.pdf"))
    plt.close(figura)


def main():
    os.makedirs(CARPETA, exist_ok=True)
    local = medianas(os.path.join("..", "resultados_v2.csv"), "principal")
    unNodo = medianas("resultados_khipu_unnodo.csv")
    dosNodos = medianas("resultados_khipu_dosnodos.csv")
    figuraSpeedupKhipu(local, unNodo)
    figuraNodos(unNodo, dosNodos)
    print("Figuras de Khipu guardadas en", CARPETA)


if __name__ == "__main__":
    main()
