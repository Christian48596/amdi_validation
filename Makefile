.PHONY: install test structural development publication clean-results

install:
	pip install -e .

test:
	pytest -q

structural:
	python experiments/run_all.py

development:
	python experiments/run_all.py --extended --sweep-budget 48

publication:
	python experiments/run_publication.py --sweep-budget 192

clean-results:
	find results -mindepth 1 ! -name .gitkeep -exec rm -rf {} +
