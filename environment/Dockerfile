# hash:sha256:0c14a334ce5f3efb9d7d68087ae7b5eeddeeadf8a59cc4d3bc4ce3474e48cb1c
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    s3fs \
    seaborn==0.13.2 \
    spikeinterface[full,widgets]==0.103.0 \
    wavpack-numcodecs==0.2.2
