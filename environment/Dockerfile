# hash:sha256:e4fabf3e0a0b6b79ac10d99ac81e7865d4bb6b4de8cf6f7b03b5550897393219
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    seaborn==0.13.2 \
    spikeinterface[full,widgets]==0.102.2 \
    wavpack-numcodecs==0.2.2

COPY postInstall /
RUN /postInstall
