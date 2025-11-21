# hash:sha256:e75e9c41ec8ef7340c466d592004e691e226cb16be85391574d70f54c524d667
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    seaborn==0.13.2 \
    spikeinterface[full,widgets]==0.103.0 \
    wavpack-numcodecs==0.2.2
