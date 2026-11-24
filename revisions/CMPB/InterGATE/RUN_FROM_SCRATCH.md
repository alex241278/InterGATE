# Ejecución desde cero — InterGATE V8.1

Extraiga este paquete en una carpeta nueva. Puede reutilizar las matrices de
entrada ya preparadas, pero no copie `artifacts/`, `results_v8/` ni checkpoints
de la primera V8. El protocolo V8.1 los rechazará por fingerprint, y mantener
los árboles separados simplifica la auditoría.

## 1. Entorno

Se recomienda Python 3.11 y una GPU NVIDIA con CUDA. En Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Para CUDA 11.8, instale primero PyTorch con el índice oficial compatible con
su sistema. Por ejemplo, para PyTorch 2.7.1:

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --user --name intergate-v8 --display-name "InterGATE V8"
```

Si ya dispone de PyTorch/CUDA funcionales, omita únicamente la primera línea
de instalación de PyTorch.

Referencias de instalación: [versiones anteriores de PyTorch](https://pytorch.org/get-started/previous-versions/)
y [guía oficial de PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/2.7.0/install/installation.html).

Antes de descargar desde NCBI/GEO, defina un correo de contacto en la sesión:

```bash
export NCBI_EMAIL="su_correo@institucion.org"
```

## 2. Recursos de datos

Los notebooks de preparación se ejecutan en este orden:

1. `data_prepare/00a_download_microarrays.ipynb`
2. `data_prepare/00b_download_rnaseq.ipynb`
3. `data_prepare/01_union_datasets.ipynb`
4. `data_prepare/02_preparacion_y_suelo.ipynb`
5. `data_prepare/03_split_y_mcombat_ref.ipynb`
6. `data_prepare/04_qc_batch_y_fuga.ipynb`
7. `data_prepare/05_supplementary_figure_mcombat_qc.ipynb`

El resultado obligatorio es `data_prepare/output_mcombat/`, con las matrices
de desarrollo y test externo separadas. La transformación M-ComBat se ajusta
en desarrollo y se proyecta sobre el externo sin refit.

Los recursos de DepMap y MSigDB se describen en
`resources_biological/README.md`.

## 3. Flujo V8

Abra Jupyter desde la raíz del proyecto:

```bash
jupyter lab
```

Ejecute, por orden:

1. `00_comprobar_instalacion.ipynb`
2. `01_validar_runtime_y_evidencia.ipynb`
3. `02_entrenar_INTERGATE_V8.ipynb`
4. `03_estabilidad_y_auditoria_V8.ipynb`

En este punto ya existen los resultados necesarios para la primera comparación
V7/V8. Genere el paquete ligero de revisión:

```bash
python tools/export_v8_review_bundle.py
```

El archivo generado se llama `InterGATE_V8_1_review_bundle.zip`. Envíelo para
revisar la comparación antes de ejecutar el notebook 04.

La comparación se hará con
`V7_V8_COMPARISON_PLAN.md`, emparejando las mismas cinco semillas y sin usar
ninguna métrica externa.

No ejecute todavía los siguientes cuadernos. Si V8.1 continúa como candidata
después de revisar 02–03, ejecute:

5. `04_baselines_y_ablacion_minima.ipynb`
6. `05_validacion_biologica_y_depmap.ipynb`
7. `06_congelar_protocolo_externo.ipynb`

El notebook 04 es el bloque más costoso porque reentrena las ablaciones
confirmatorias para las cinco semillas. Puede interrumpirse y reanudarse.

## 4. Test externo

No ejecute `07_test_externo_bloqueado.ipynb` para elegir entre V7 y V8. Debe
abrirse únicamente después de fijar la versión final, los hiperparámetros y
las reglas de análisis. El notebook comprueba el lock SHA-256 y se niega a
sobrescribir una evaluación ya completada.

El bloqueo es una garantía del flujo limpio actual, no una afirmación de que
las cohortes nunca se hayan explorado históricamente. Cualquier exploración
externa anterior debe quedar declarada al interpretar el manuscrito.

## 5. Reconstrucción completa del manuscrito si V8 es la elegida

Los siguientes cuadernos son posteriores a la decisión V7/V8. No los use para
escoger versión:

1. `09_ablaciones_biologia_score_0075.ipynb` — solo desarrollo; puede
   ejecutarse inmediatamente después del lock 06, antes o después del 07.
2. `07_test_externo_bloqueado.ipynb` — primera apertura del externo y única
   evaluación predictiva del modelo.
3. `08a_benchmark_tabular_pareado_S12.ipynb` — benchmark predictivo y S12;
   queda declarado como análisis post hoc de protocolo fijo.
4. `08b_concordancia_estructural_externa.ipynb` — transferencia estructural,
   sin refit ni nueva selección del grafo.
5. `12_controles_revision_q_permutaciones.ipynb` — factorial InterGATE/q,
   sensibilidad k/R, auditoría numérica y 49 permutaciones reentrenadas.
6. `13_generate_supplementary_figures_S2_S3.ipynb` — genera S2/S3 desde las
   tablas del 12, sin recalcular estadísticas.
7. `10_figuras_definitivas.ipynb` — genera al final las cinco figuras
   principales desde las tablas V8.

El cuaderno 10 se ejecuta al final aunque conserve su número histórico. No hay
cuaderno 11 en esta serie; la numeración se mantiene para corresponder con el
material del manuscrito.

## 6. Reanudación segura

`RESUME=True` no mezcla versiones. Antes de reutilizar un checkpoint se
comprueban:

- versión y protocolo V8;
- configuración completa de arquitectura y entrenamiento;
- código que define entrenamiento y propagación;
- brazo de ablación efectivo;
- matrices de expresión, características y grafo candidato;
- etiquetas, cohortes y splits.

Si cualquiera de estos elementos cambia, el proceso se detiene y exige una
ruta nueva. No copie dentro de este proyecto carpetas `artifacts` o
`results_v8` procedentes de otras versiones.
