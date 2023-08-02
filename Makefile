# Makefile to set up virtual environment and install required packages


venv: 
	python3.10 -m venv h2o

run:
	./h2o/bin/wave run --no-reload src.app

setup: venv
	mkdir datasets
	mkdir inference_data
	./h2o/bin/python3.10 -m pip install tabulate \
		future \
		matplotlib \
		h2o-wave \
		pandas
	./h2o/bin/python3.10 -m pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o

.PHONY: venv run setup