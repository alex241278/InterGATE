# InterGATE

InterGATE learns subtype-associated gene networks from breast cancer expression data and a candidate interaction network. The `intergate/` directory contains the Python implementation. Model settings are supplied in `configs/` and the locked benchmark specifications in `protocols/`.

The archive's [source and provenance guide](../README.md) gives the software versions, data sources, checksums and instructions for reproducing the additional analyses. The [cohort preparation sequence](data_prepare/ORDEN_EJECUCION.md) and the numbered notebooks in `notebooks/` cover data preparation, training and evaluation. The [methods description](METHODS_V8.md) specifies the model. The 50 symbols in `resources_biological/PAM50_symbols.txt` are the exclusion list used in the study's signature analyses.
