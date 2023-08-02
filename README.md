# Nocode ML Using H20 AutoML and H2O Wave

## Video Demo
https://github.com/Neethamadhu-Madurasinghe/nocodeML/assets/87432896/c5fde232-5589-4499-acea-f9315c29094e


## Getting Started
Tested on Python 3.10

1. Clone the repo

```sh
git clone https://github.com/Neethamadhu-Madurasinghe/nocodeML.git
```

2. setup environment

```sh
cd nocodeML
make setup
```

3. Run the app

```sh
make run
```

This will start the app at <http://localhost:10101/#import>.

## Docker

1. Build docker image
```sh
sudo docker build --platform linux/x86_64 --build-arg PYTHON_VERSION=3.10.10 --build-arg WAVE_VERSION=0.26.1 --build-arg PYTHON_MODULE="src.app" -t nocodeml:0.1.0 .
```
2. Run
```sh
sudo docker run --rm --name nocodeml -p 10101:8080 -e PORT=8080 nocodeml:0.1.0
```

This will start the app at <http://localhost:10101/#import>.

## Usecases

This app can be used to quicky train and use Regression and classification models. Supports both real time and batch predictions

## Learn More

To learn more about H2O AutoML, check out the [docs](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html).

To learn more about H2O Wave, check out the [docs](https://wave.h2o.ai/).

