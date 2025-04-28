# hash:sha256:93d9929765b702d4addf1044994d75a51c127b1bd6bb4edda81fa5a1b41dbf4e
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
