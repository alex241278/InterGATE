PYTHON ?= python

.PHONY: install preflight-data preflight-analysis data analysis all test

install:
	$(PYTHON) -m pip install -e ".[full]"

preflight-data:
	$(PYTHON) scripts/preflight.py --stage data

preflight-analysis:
	$(PYTHON) scripts/preflight.py --stage analysis

data:
	$(PYTHON) scripts/run_pipeline.py data

analysis:
	$(PYTHON) scripts/run_pipeline.py analysis

all:
	$(PYTHON) scripts/run_pipeline.py all

test:
	$(PYTHON) -m compileall -q intergate data_prepare scripts tests
	$(PYTHON) tests/check_notebook_contract.py
	$(PYTHON) tests/test_protocol_alignment.py
	$(PYTHON) tests/test_v6_synthetic.py

