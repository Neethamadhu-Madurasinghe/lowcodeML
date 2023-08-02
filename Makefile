# Makefile to set up virtual environment and install required packages

setup:
	mkdir datasets
	mkdir inference_data
	virtualenv h2o
	sudo apt install python3-virtualenv
	. h2o/bin/activate && \
		pip install tabulate && \
		pip install future && \
		pip install matplotlib && \
		pip install h2o-wave && \
		pip install pandas && \
		pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o

.PHONY: setup
