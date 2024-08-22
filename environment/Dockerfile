# hash:sha256:54723249b683b9768ddb3e3b04d2f7c13c04bc930ddfcdd7e064abde360c7572
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    seaborn==0.13.2 \
    spikeinterface[full,widgets]==0.101.0 \
    wavpack-numcodecs==0.1.5

COPY postInstall /
RUN /postInstall
