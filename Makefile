.PHONY: install data check smoke toy notebooks backbones clean

install:
	pip install -e ".[all]"

data:
	python scripts/download_zenodo_data.py --extract

check:
	python scripts/00_check_setup.py

smoke:
	python scripts/01_smoke_test.py

toy:
	python scripts/02_run_toy_example.py

notebooks:
	bash scripts/run_notebooks.sh

backbones:
	bash scripts/run_backbone_ablation.sh

clean:
	rm -rf __pycache__ intergate/__pycache__ .pytest_cache
