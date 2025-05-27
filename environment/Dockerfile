# hash:sha256:612df1e256b2cd929918cf5e58c49d4b4ee0a1af169f6521bdd08ca365dd6806
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

COPY postInstall /
RUN /postInstall
