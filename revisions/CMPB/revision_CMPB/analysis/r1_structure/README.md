# Sensibilidad de la red Basal-like

La rejilla y los controles de esta carpeta son análisis posteriores a la recepción del informe. El programa compara cada reexportación con los cinco paneles enviados y exige igualdad de las 529 aristas por semilla y de las 299 que forman su intersección. Las asignaciones de aristas, sus genes y las tablas sin filtrar quedan disponibles en los ficheros comprimidos.

| Análisis | Programa | Resultado principal |
|---|---|---|
| Pesos, selector, tríos, máscaras y prior fijo | `structural_sensitivity.py` | `structure_summary.csv`, `structure_core_edges.csv.gz`, `structure_seed_panels.csv.gz`, `prior_random_tie_audit.csv.gz` |
| Cobertura física y controles sobre otro prior | `string_prior.py` | `string_coverage_vs_matched_prior.csv`, `string_independent_rankings.csv`, `string_core_coverage.csv`, `string_selected_pairs.csv.gz`, `string_manifest.json` |
| Reentrenamiento oficial de las semillas 10001 a 10015 | `train_additional_seeds.py` | `training/artifacts/INTERGATE_REVISION_SEEDS/seed_*/` |
| Agregación en cuanto se completan las semillas 10001–10005 y 10001–10015 | `aggregate_when_ready.py` | `extended_10_{summary,core_edges,seed_panels}` y `extended_20_{summary,core_edges,seed_panels}` con sus extensiones CSV o CSV.GZ |

## Ejecución

Desde esta carpeta y con el paquete InterGATE en su ubicación relativa original se pueden recalcular los resultados con estos comandos. El directorio de STRING debe contener los dos archivos gzip públicos identificados por nombre, URL y SHA-256 en `string_manifest.json`.

```bash
python -B structural_sensitivity.py
python -B string_prior.py /ruta/a/STRING_v12.0
python -Bu train_additional_seeds.py
python -B structural_sensitivity.py --extended 10
python -B structural_sensitivity.py --extended 20
```

La opción `--extended 10` requiere que estén disponibles las cinco primeras semillas adicionales y `--extended 20` las quince. Para generar los resultados automáticamente, se puede iniciar `python -Bu aggregate_when_ready.py` en otro proceso una vez arrancado el entrenador. Antes de comenzar un ajuste, se cotejan los hashes de datos, configuración e implementación y los índices de entrenamiento y validación con el bundle original. El entrenamiento escribe en una ruta propia, usa un bloqueo de proceso y reutiliza los checkpoints compatibles. La agregación de las nuevas semillas utiliza únicamente el universo candidato y los resultados de desarrollo.

La comparación estática de 529 aristas usa los efectos externos ya calculados sobre todos los candidatos. Las filas producidas con `--extended` indican `external_evaluated=False`. Las columnas vacías del nulo se explican en `null_infeasible_strata` indicando tipo, signo, estratos de grado y el recuento solicitado/disponible. El CSV recoge la media y el error estándar Monte Carlo del nulo, además de las fracciones y razones observadas. Las proporciones de recuperación se calculan sobre aristas dirigidas o genes distintos, según indica la columna. Los controles STRING trabajan con pares físicos sin dirección. Las dos máscaras de fuente se aplican a la exportación del modelo entrenado sobre la unión HuRI--OmniPath.

El cuaderno original de instalación registró Python 3.11.9, PyTorch 2.7.1+cu118, NumPy 2.4.4, pandas 2.3.3, SciPy 1.17.1 y scikit-learn 1.9.0. Para las semillas adicionales se usó el entorno Torch311 con Python 3.11.9, PyTorch 2.7.1+cu118, NumPy 2.3.5, pandas 2.3.3, SciPy 1.17.1 y scikit-learn 1.8.0. La RTX 3090 y las versiones del entorno de ajuste también se identifican en `training/seed_contract.json`. El análisis de pesos y el muestreo de nulos se ejecutaron con NumPy 2.5.3, pandas 3.0.6 y SciPy 1.18.1; el comparador STRING utilizó Torch311. El acuerdo exacto de las aristas reexportadas en las cinco semillas se verificó antes de analizar las variantes.

## Reentrenamiento bajo STRING

El control de sustituir el prior completo se ejecuta con `train_string_prior.py`. El modo `preflight` verifica ambas huellas de los datos públicos, recupera el hash de la matriz original con el runtime oficial, conserva las mismas 7.131 columnas y cambia el prior a STRING físico con puntuación mínima 700. Las 28.447 interacciones no dirigidas se expanden sobre 56.894 arcos orientados por orden de los identificadores. Puesto que el selector retiene arcos individuales, el modelo usa el override oficial de 529 aristas (0,929799 %). Se recalculan el contexto por subtipo y las covariables del grafo. El contrato registra las dimensiones, la identidad de pacientes y genes, los seis campos de configuración que cambian y la verificación de una pasada por el modelo sobre un tumor de entrenamiento. Los coeficientes del score en la agregación final son los enviados en el manuscrito.

```bash
python -Bu train_string_prior.py preflight /ruta/a/STRING_v12.0
python -Bu train_string_prior.py queued-training /ruta/a/STRING_v12.0
python -B analyze_string_fits.py
```

El segundo comando espera los quince puntos finales del estudio de semillas, adquiere el bloqueo de la GPU compartido con ese entrenador y ejecuta el flujo estructural, de soporte y de clasificación para cada una de las cinco semillas originales. Cuando completa los cinco ajustes, invoca por sí mismo el tercer programa. Este genera en `string_training/` el resumen completo de la intersección, aristas de cada semilla y del núcleo, coincidencia en aristas y pares físicos con el núcleo HuRI--OmniPath, enriquecimiento y macro-F1 de validación bajo el prior sustitutivo.

## Continuidad durante esta revisión

La unidad transitoria de usuario `intergate-cmpb-r1-20261123.service` supervisa los tres procesos una vez cerrado el chat. El estado se actualiza cada 45 segundos en `supervision/status.json`. Cada trabajador ya activo conserva el bloqueo de su etapa. Si queda libre un bloqueo y faltan resultados, el supervisor inicia el programa correspondiente con un máximo de cuatro intentos automáticos por etapa y una duración total de 72 horas. El programa de ajuste recupera el modelo, el optimizador y los estados aleatorios del último episodio completo tras verificar los hashes. Si se interrumpe una fase posterior al último checkpoint estructural, se repite la parte pendiente de la semilla. Los ajustes completos se detectan por sus marcadores.

`systemctl --user status intergate-cmpb-r1-20261123.service` muestra el estado del supervisor. El expediente conserva los checkpoints después de un reinicio del equipo. Para retomar entonces la supervisión dentro de la ventana original, puede ejecutarse `supervise_revision.py` con el intérprete Torch311, la ruta a los datos STRING y la ruta al intérprete utilizado para la agregación, o reanudar los tres comandos de cada apartado. La unidad transitoria finaliza al completarse las tres etapas.
