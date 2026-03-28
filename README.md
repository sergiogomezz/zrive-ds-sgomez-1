# Zrive Applied Data Science Program

Work completed as part of the [Zrive Applied Data Science](https://zriveapp.com/cursos/zrive-applied-data-science) program,
a hands-on bootcamp focused on building end-to-end ML pipelines following industry best practices.

Topics covered: data ingestion, API integration, preprocessing, feature engineering,
model training, evaluation, and production-ready code standards (testing, linting, CI).

## Setup

Python 3.11.0 via `pyenv` + `poetry` for dependency management.
```bash
pyenv install 3.11.0
poetry install
```

## Usage
```bash
make lint   # format with black
make test   # lint + pytest
```
