# hash:sha256:8e3e4633ed4747822d39a5e988b793ad99a592c220a0891424cb54f7ae981cce
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    s3fs==2026.4.0 \
    seaborn==0.13.2 \
    spikeinterface[full,widgets]==0.104.2 \
    wavpack-numcodecs==0.2.2
