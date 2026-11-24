# Recursos biológicos externos

El notebook 05 crea esta estructura:

```text
resources_biological/
├── msigdb/
│   ├── h.all.v2026.1.Hs.symbols.gmt
│   └── c2.cgp.v2026.1.Hs.symbols.gmt
└── depmap/
    ├── CRISPRGeneEffect.csv
    ├── Model.csv
    └── OmicsExpressionProteinCodingGenesTPMLogp1.csv   # si hace falta
```

MSigDB puede descargarse desde el propio notebook. Si Broad cambia el nombre
de la versión, descargue los GMT de símbolos Hallmark y C2:CGP y ajuste solo
las dos rutas declaradas al principio del notebook antes de congelar el
protocolo.

DepMap no se redistribuye en este paquete. Coloque una release pública
coherente y no mezcle versiones. Se aceptan los dos nombres habituales de la
matriz de expresión de genes codificantes. Los hashes, release y método de
agrupación quedan guardados en el manifiesto.

Los grupos celulares son proxies: `receptor_positive`, `her2_amplified` y
`receptor_negative`. Si `Model.csv` no contiene una anotación molecular que
cubralos, el notebook deriva los grupos con ESR1/PGR/ERBB2 y registra todos
los umbrales. Esto no convierte líneas celulares en tumores PAM50.
