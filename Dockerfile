FROM python:3.9-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /src

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && apt-get install -y gcc python3-dev \
    && apt-get install -y curl \
    && apt-get purge -y --auto-remove

RUN curl https://sdk.cloud.google.com > install.sh \
    && bash install.sh --disable-prompts

ENV PATH $PATH:/root/google-cloud-sdk/bin

COPY CHANGELOG.md LICENSE README.md batch.sh setup.py ./
COPY src src

RUN pip install --upgrade pip && \
    pip install -e '.'

ENTRYPOINT ["/bin/bash"]
