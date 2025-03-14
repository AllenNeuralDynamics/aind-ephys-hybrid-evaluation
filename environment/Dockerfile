# hash:sha256:430523a86a972d8eb3f4aad80fcdc09e03d6a7222649c00ef2ee34ff957e2c94
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    seaborn==0.13.2 \
    spikeinterface[full,widgets]==0.102.1 \
    wavpack-numcodecs==0.2.2

COPY postInstall /
RUN /postInstall
