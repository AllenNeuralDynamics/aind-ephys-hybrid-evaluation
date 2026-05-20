# hash:sha256:d55af04ded55c9f7981275678a4f24c61bf0b9a91db8d53578886330ce656924
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y \
        jupyterlab==4.5.7 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    s3fs==2026.4.0 \
    seaborn==0.13.2 \
    spikeinterface[full,widgets]==0.104.2 \
    wavpack-numcodecs==0.2.2
