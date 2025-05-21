# hash:sha256:97ecc515f6f811b13e784ff3fb1ee69bcdba9337987c1cdd3bfbc3198aa621db
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    seaborn==0.13.2 \
    spikeinterface[full,widgets]==0.102.3 \
    wavpack-numcodecs==0.2.2
