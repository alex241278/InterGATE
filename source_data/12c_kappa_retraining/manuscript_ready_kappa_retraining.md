# Sensibilidad de κ con reentrenamiento

Se reentrenó el flujo completo en cinco objetivos de sparsidad (κ=0.005–0.020; K=264–1058 aristas por clase) manteniendo constantes semillas, particiones, presupuesto de optimización, pérdidas, arquitectura y métrica de checkpoint. El modelo principal κ=0.010 se reutilizó y los cuatro valores alternativos aportaron veinte entrenamientos nuevos. Todas las métricas proceden de desarrollo-validación.

El macro-F1 medio varió entre 0.888 y 0.890. El diagnóstico pareado de no inferioridad con margen −0.01 fue compatible en las cuatro densidades alternativas.

En las redes Basal-like, el Jaccard medio entre semillas fue 0.478–0.677, la estabilidad de Nogueira 0.644–0.803 y la fracción del núcleo 5/5 0.356–0.630. El coeficiente de solapamiento de los consensos alternativos con κ=0.010 fue 0.919–0.964. La recuperación Basal-like permaneció en 1.84–2.22 veces la expectativa emparejada para al menos un extremo y 3.99–6.57 para ambos extremos. El margen de especificidad fue 0.001–1.480 log2-fold.

Conclusión del análisis: recuperación biológica robusta=sí; dependencia positiva de GNN y message passing en toda la rejilla=sí; diagnóstico predictivo pareado compatible con el margen predefinido=sí. κ=0.010 permanece como especificación principal y estos resultados se interpretan exclusivamente como sensibilidad, no como una nueva selección de hiperparámetros.
