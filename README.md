# Nocode ML Using H20 AutoML and H2O Wave

## Video Demo

## Getting Started
Tested on Python 3.10.10

1. Clone the repo

```sh
git clone https://github.com/Neethamadhu-Madurasinghe/nocodeML.git
```

2. Execute make file

```sh
cd nocodeML
make setup
```

3. Switch to the python virtual environment

```sh
source h2o/bin/activate
```

4. Run the app

```sh
wave run src.app
```

This will start the app at <http://localhost:10101>.

## Docker

1. Build docker image
```sh
sudo docker build --platform linux/x86_64 --build-arg PYTHON_VERSION=3.10.10 --build-arg WAVE_VERSION=0.26.1 --build-arg PYTHON_MODULE="src.app" -t nocodeml:0.1.0 .
```
2. Run
```sh
sudo docker run --rm --name nocodeml -p 10101:8080 -e PORT=8080 nocodeml:0.1.0
```

This will start the app at <http://localhost:10101>.

## Usecases

This app can be used to quicky train and use Regression and classification models. Supports both real time and batch predictions

## Learn More

To learn more about H2O AutoML, check out the [docs](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html).

To learn more about H2O Wave, check out the [docs](https://wave.h2o.ai/).

