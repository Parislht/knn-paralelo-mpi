# KNN paralelo con MPI

Proyecto del curso **CS4052 Computación Paralela y Distribuida** (UTEC, ciclo 2026-II).

**Integrantes:** Ary Werner Aaron Rojas Durand, Paris Lenard Herrera Torres y Joel David Miguel Fernández.

## De qué trata

El proyecto paraleliza un clasificador KNN (*k-nearest neighbors*) sobre el conjunto `load_digits` de scikit-learn: 1797 imágenes de dígitos de 8×8 píxeles, en las que cada imagen es un punto de 64 dimensiones. Partimos del código secuencial compartido en el enunciado y lo llevamos a memoria distribuida con `mpi4py`.

En KNN, cada punto de prueba se compara contra todos los puntos de entrenamiento, así que el trabajo se puede ver como el llenado de una matriz de distancias de tamaño `n_te × n_tr`. Cada distancia es independiente de las demás; la única dependencia aparece al final, cuando para cada consulta hay que elegir sus `k` vecinos más cercanos y votar.

Paralelizar KNN es, básicamente, decidir cómo cortar esa matriz entre los procesos. Organizamos los procesos en una rejilla `pr × pc`, y de ahí salen tres estrategias:

- **A (`pr = 1`):** se reparte el entrenamiento y se difunde la prueba. Es la estrategia del DAG del enunciado.
- **B (`pc = 1`):** se reparte la prueba y se difunde el entrenamiento.
- **C (rejilla intermedia):** se reparten ambos conjuntos en bloques.

Las tres usan las mismas operaciones colectivas (`scatter`, `bcast` y `gather`) y salen del mismo programa, cambiando solo `pr` y `pc`.

## Versiones beta

El desarrollo se hizo en tres versiones, que quedan como archivos separados:

1. **`knn_v1.py`:** estrategia A tal cual el DAG del enunciado, con `comm.scatter`, `comm.bcast` y `comm.gather`, y la fusión de candidatos en la raíz.
2. **`knn_v2.py`:** estrategias A y B con el cálculo de distancias vectorizado en NumPy y la medición separada de los tiempos de cómputo y de comunicación.
3. **`knn_v3.py`:** la rejilla general `pr × pc` con subcomunicadores (`comm.Split`). Es la versión con la que se hicieron todas las mediciones.

Las tres producen exactamente las mismas predicciones que el código del enunciado (ver `validacion.txt`).

## Archivos

| Archivo | Contenido |
|---|---|
| `knn_enunciado.py` | Código secuencial compartido en el enunciado (sin el gráfico) |
| `knn_v1.py` | Versión beta 1: estrategia A según el DAG (`scatter`, `bcast`, `gather`) |
| `knn_v2.py` | Versión beta 2: estrategias A y B vectorizadas, con tiempos de cómputo y comunicación |
| `knn_v3.py` | Versión beta 3: rejilla general `pr x pc` (A con `pr=1`, B con `pc=1`) |
| `knn_comun.py` | Funciones compartidas: distancias por trozos, top-k local, fusión y voto |
| `generar_datos.py` | Aumento de `load_digits` hasta N = 32 000 |
| `validar.py` | Compara las predicciones de todas las versiones con el código del enunciado |
| `barrido.py` | Ejecuta todas las configuraciones y guarda los tiempos en un CSV |
| `graficas.py` | Ajusta las constantes del modelo y genera las figuras del informe |
| `resultados_v2.csv`, `validacion.txt` | Resultados medidos (AMD Ryzen 7 5800XT, 8 núcleos) |

## Datos

Como `load_digits` es pequeño, `generar_datos.py` lo aumenta con variaciones de los dígitos reales: desplazamientos de hasta un píxel y un poco de ruido, conservando la etiqueta de cada imagen. Para no mezclar información entre entrenamiento y prueba, primero se parten los 1797 dígitos originales con la misma semilla del enunciado (`test_size=0.2`, `random_state=42`) y después se aumenta cada lado por separado. Se generan conjuntos de 2 000 a 32 000 muestras, además de dos con la proporción entre entrenamiento y prueba invertida.

## Cómo ejecutarlo

**Requisitos:** una implementación de MPI (en Windows, MS-MPI 10.1.3) y Python 3.10 o superior con `mpi4py`, `numpy`, `scikit-learn`, `pandas` y `matplotlib`.

```
python validar.py
python generar_datos.py
python barrido.py
python graficas.py resultados_v2.csv
```

1. `validar.py` corre el código del enunciado y las tres versiones con 1, 4 y 8 procesos, y comprueba que las predicciones coincidan.
2. `generar_datos.py` crea la carpeta `datos/` con los conjuntos aumentados.
3. `barrido.py` ejecuta las 64 configuraciones (tamaños, número de procesos y rejillas), cada una con una corrida de calentamiento y tres corridas medidas, y guarda los tiempos en `resultados_v2.csv`. Si se interrumpe, al volver a ejecutarlo continúa donde se quedó. Con `--rapido` corre solo una prueba corta.
4. `graficas.py` ajusta las constantes del modelo de costos y genera las figuras del informe en la carpeta `figuras/`.

También se puede correr una configuración suelta, por ejemplo la rejilla 2×4 con 8 procesos:

```
mpiexec -n 8 python knn_v3.py --datos datos/N16000_ts20.npz --pr 2 --pc 4
```

Cada script limita a NumPy a un hilo por proceso, para que el paralelismo medido sea solo el de MPI.

## Resultados principales

- Todas las versiones y rejillas dan exactamente las mismas predicciones que el código del enunciado.
- La vectorización con NumPy hace el cálculo unas 24 veces más rápido que el código del enunciado, antes de paralelizar. Por eso el speedup se mide contra la versión vectorizada con un proceso.
- La comunicación pesa menos del 1 % del tiempo con los tamaños grandes, y el modelo de costos del curso la describe bien.
- Con `n_tr > n_te`, la estrategia A comunica menos que B; al invertir la proporción gana B. Es decir, conviene la estrategia que no replica el conjunto grande.
- En una sola máquina, el speedup llega a ~2.1 con 4 procesos y no mejora con 8. La pérdida está en el cómputo y no en la comunicación: el tiempo por operación crece cuando varios procesos trabajan a la vez, lo que creemos que se debe a que comparten la caché y la memoria del equipo.

El análisis completo está en el informe del proyecto.
