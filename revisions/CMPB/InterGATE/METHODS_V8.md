# Nota metodológica de InterGATE V8.1

## Diseño preservado

El endpoint contiene LumA, LumB, HER2-enriched y Basal-like. Las cohortes de
desarrollo son GSE96058, GSE25066, GSE162228 y GSE20711; TCGA-BRCA, GSE21653
y GSE1456 forman el bloque externo. Se utilizan cinco semillas
preespecificadas: 1234, 2025, 7, 99 y 31337.

La fase 1 aprende grafos por clase bajo un prior HuRI+OmniPath, presupuesto
Exact-K y metaepisodios leave-one-development-cohort-out. La fase 2A entrena la
GNN con grafo fijo; la fase 2B fusiona GNN, `X_h` y una rama tabular. Las
anotaciones PAM50, MSigDB y DepMap no participan en el ajuste.

## Cambios V8

1. Un embedding entrenable identifica cada gen. Es apropiado para un grafo
   biológico con nodos semánticamente fijos, pero puede aportar señal aun si
   las aristas se alteran.
2. Cada experto recibe una pérdida auxiliar de clasificación de cuatro clases
   con peso 0.20. La supervisión profunda favorece representaciones
   intermedias discriminativas [1].
3. La rama tabular recibe el universo completo de genes conectados y utiliza
   anchura 256.
4. Durante fase 2B se elimina la rama tabular en el 30 % de las muestras. El
   dropout de modalidades busca evitar el colapso hacia una sola rama [2–4].
5. Los logits GNN reciben una pérdida auxiliar con peso 0.50.
6. El microbatch de fase 1 baja de 4 a 2 por memoria, la acumulación sube de
   60 a 120 y los pasos por episodio suben de 96 a 192. Así se conservan los
   384 ejemplos por flujo y las dos actualizaciones de V7 (240 + 144). Los
   lotes de soporte suben de 32 a 64 para conservar 128 ejemplos por paso. El
   cambio de microbatch se declara porque puede alterar ligeramente la
   dinámica de BatchNorm aun con el presupuesto total pareado [9,10].

## Lectura de las ablaciones

La anulación post hoc de una rama mide dependencia del modelo ya entrenado,
no información condicional recuperable tras reentrenamiento. Además, el
dropout es asimétrico: la ausencia de la rama tabular se ve durante el ajuste,
pero la ausencia de la GNN no. Por ello las caídas de ambas ramas no constituyen
una escala causal comparable.

La importancia de la topología exige controles con los mismos nodos, K,
grados, tipos y signos: grafo aprendido, rewired, prior aleatorio emparejado y
modelo sin message passing. Los baselines independientes de estructura son
necesarios para atribuir una ventaja específicamente al grafo [5]. El borrado
seguido de reentrenamiento evita parte del cambio de distribución inducido por
una ablación puramente post hoc [6].

## Controles y salidas del manuscrito

La validación biológica conserva las firmas direccionales corregidas del
manuscrito V7 y excluye PAM50 del análisis primario. Así, cualquier diferencia
V7/V8 procede del modelo y sus redes, no de cambiar la anotación de referencia.
Los análisis post hoc incluyen 1.000 paneles candidatos emparejados, 1.000
rewirings dirigidos que preservan grado/tipo/signo, el factorial InterGATE/q ×
selección raw/modular, sensibilidad k/R y 49 permutaciones de etiquetas con
reentrenamiento completo en dos semillas. Estos controles no se utilizan para
seleccionar entre V7 y V8.

La evaluación predictiva externa se ejecuta una sola vez en el notebook 07.
Después, el 08b puede volver a leer las mismas matrices para estimar la
concordancia estructural del núcleo ya bloqueado, sin refit, reselección ni
modificación de hiperparámetros. El benchmark tabular pareado se declara
`posthoc_fixed_protocol`.

## Referencias

1. Lee CY, Xie S, Gallagher P, Zhang Z, Tu Z. Deeply-Supervised Nets. *AISTATS*.
   2015;38:562–570. https://proceedings.mlr.press/v38/lee15a.html
2. Neverova N, Wolf C, Taylor GW, Nebout F. ModDrop: adaptive multi-modal
   gesture recognition. *IEEE TPAMI*. 2016;38:1692–1706.
   https://doi.org/10.1109/TPAMI.2015.2461544
3. Wang W, Tran D, Feiszli M. What Makes Training Multi-Modal Classification
   Networks Hard? *CVPR*. 2020:12695–12705.
   https://doi.org/10.1109/CVPR42600.2020.01271
4. Peng X, Wei Y, Deng A, Wang D, Hu D. Balanced Multimodal Learning via
   On-the-Fly Gradient Modulation. *CVPR*. 2022:8238–8247.
   https://doi.org/10.1109/CVPR52688.2022.00806
5. Errica F, Podda M, Bacciu D, Micheli A. A Fair Comparison of Graph Neural
   Networks for Graph Classification. *ICLR*. 2020.
   https://openreview.net/forum?id=HygDF6NFPB
6. Hooker S, Erhan D, Kindermans PJ, Kim B. A Benchmark for Interpretability
   Methods in Deep Neural Networks. *NeurIPS*. 2019;32.
   https://arxiv.org/abs/1806.10758
7. Parker JS, Mullins M, Cheang MCU, et al. Supervised Risk Predictor of Breast
   Cancer Based on Intrinsic Subtypes. *J Clin Oncol.* 2009;27:1160–1167.
   https://doi.org/10.1200/JCO.2008.18.1370
8. Tsherniak A, Vazquez F, Montgomery PG, et al. Defining a Cancer Dependency
   Map. *Cell.* 2017;170:564–576.e16.
   https://doi.org/10.1016/j.cell.2017.06.010
9. Goyal P, Dollár P, Girshick R, et al. Accurate, Large Minibatch SGD:
   Training ImageNet in 1 Hour. 2017. https://arxiv.org/abs/1706.02677
10. Smith SL, Kindermans PJ, Ying C, Le QV. Don't Decay the Learning Rate,
    Increase the Batch Size. *ICLR*. 2018.
    https://openreview.net/forum?id=B1Yy1BxCZ
