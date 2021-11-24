FROM python:3.8-slim

ENV POETRY_VIRTUALENVS_CREATE=false

RUN apt-get update \
    && apt-get install -y git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir poetry
