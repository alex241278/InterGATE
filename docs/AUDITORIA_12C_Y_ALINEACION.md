# Auditoría de 12c y alineación manuscrito–código

Fecha: 30 de agosto de 2026

## Conclusión

La justificación de \(\kappa=0.010\) es metodológicamente plausible y está
respaldada por el reentrenamiento completo. No debe describirse como el valor
que maximiza todas las métricas: las redes más densas son más estables. La
lectura sustentada por los resultados es que el 1% constituye un compromiso
parsimonioso y el principal punto de transición entre estabilidad, cohesión y
tamaño de red, conservando la recuperación Basal-like y el rendimiento
predictivo.

## Integridad de la ejecución 12c

- Se evaluaron cinco densidades:
  \(\kappa=0.005,0.007,0.010,0.015,0.020\), equivalentes a
  \(K=264,370,529,793,1058\).
- Se completaron 25 combinaciones densidad–semilla: las cinco ejecuciones
  primarias en 0.010 y 20 reentrenamientos nuevos.
- En cada densidad se reentrenaron Phase 1 y Phase 2.
- La auditoría de configuración confirma que `KEEP_MIN` fue el único campo del
  modelo modificado.
- El bloque externo permaneció cerrado.
- El SHA-256 del protocolo y el fingerprint de los 15 módulos que definen el
  entrenamiento coinciden exactamente entre el manifiesto 12c y la
  instantánea de código entregada.
- Los hashes de todos los ficheros declarados en el manifiesto fueron
  verificados.

## Resultados que sustentan el presupuesto primario

| \(\kappa\) | \(K\) | Macro-F1 | Jaccard Basal | Nogueira | Strict/\(K\) | Fracción gigante | Fold Basal ≥1 extremo | Fold ambos |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.005 | 264 | 0.8880 | 0.478 | 0.644 | 0.356 | 0.147 | 2.173 | 6.573 |
| 0.007 | 370 | 0.8895 | 0.513 | 0.674 | 0.435 | 0.164 | 2.210 | 5.844 |
| **0.010** | **529** | **0.8877** | **0.607** | **0.752** | **0.565** | **0.432** | **2.220** | **5.909** |
| 0.015 | 793 | 0.8881 | 0.629 | 0.768 | 0.567 | 0.612 | 1.996 | 4.669 |
| 0.020 | 1,058 | 0.8896 | 0.677 | 0.803 | 0.630 | 0.680 | 1.842 | 3.990 |

El rendimiento predictivo varía menos de 0.002 de macro-F1 entre densidades y
todos los límites inferiores pareados quedan por encima del margen diagnóstico
de \(-0.01\). La neutralización de la GNN y la eliminación del message passing
reducen macro-F1 en las cinco densidades. Entre 0.007 y 0.010 aumenta
marcadamente la estabilidad multisemilla, la retención estricta y la cohesión;
al pasar de 0.010 a 0.015 el número de aristas crece un 50%, pero la fracción
estricta apenas cambia (0.565 a 0.567) y disminuye la recuperación Basal-like.
Las redes alternativas conservan un coeficiente de solapamiento de
0.919–0.964 con el consenso primario.

## Cambios incorporados al manuscrito

- Justificación del presupuesto Exact-K en Métodos, con el 1% descrito como
  cuota scale-free, idéntica entre clases y apta para auditoría multisemilla.
- Nueva subsección metodológica de sensibilidad mediante reentrenamiento
  completo.
- Nueva subsección de Resultados con rendimiento, dependencia del grafo,
  estabilidad, topología y recuperación Basal-like.
- Integración de la lectura de robustez en Discusión y Conclusión.
- Nueva Supplementary Table S14 con los valores completos.
- Nueva Supplementary Figure S8 y notebook 12d reproducible.
- Incorporación de la referencia metodológica de estabilidad StARS junto a la
  estabilidad de selección ya citada. StARS se usa como principio conceptual,
  no como algoritmo aplicado en este estudio.

## Auditoría editorial

- Artículo principal: 26 páginas, compilación correcta.
- Suplemento: 17 páginas, compilación correcta.
- No hay referencias, citas bibliográficas ni gráficos sin resolver.
- Todas las figuras y tablas del texto principal y del suplemento están citadas.
- No hay etiquetas duplicadas ni saltos de numeración.
- La nueva tabla y la nueva figura se añaden al final como S14 y S8, sin
  renumerar los elementos anteriores.
- Los cinco highlights siguen priorizando el hallazgo Basal-like, la
  reproducibilidad, la ventaja frente a XGBoost, la concordancia externa y la
  clasificación Basal-like.

## Auditoría del repositorio preparado

- 25 notebooks limpios, sin outputs ni contadores de ejecución.
- 316 celdas totales y 198 celdas de código compilables.
- Los módulos Python, scripts, tests y notebooks pasan compilación estática.
- El test de alineación protocolo–configuración–documentación–gráfico pasa.
- El runner reproduce el orden de dependencias e incluye 08c, 12c y 12d.
- Se incluyen el protocolo, preparación de datos, entorno, tests, documentación,
  recursos, fuente compacta de 12c y el graphical abstract.

## Único bloque pendiente

Las permutaciones de etiquetas en curso no se han usado ni anticipado en esta
edición. Cuando terminen, deben verificarse su manifiesto, completitud y familia
inferencial antes de decidir si se incorporan como control suplementario. El
resto del texto y la justificación de \(\kappa=0.010\) ya quedan sustentados por
el análisis 12c.
