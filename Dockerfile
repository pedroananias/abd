FROM python:3.9-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /src

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && apt-get purge -y --auto-remove \
    && rm -rf /var/lib/apt/lists/*

COPY CHANGELOG LICENSE README.md batch.sh setup.py ./
COPY src src

RUN pip install --upgrade pip wheel && \
    pip install .

CMD abd --help
