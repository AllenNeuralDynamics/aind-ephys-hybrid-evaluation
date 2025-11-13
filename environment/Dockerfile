# hash:sha256:3976a0572eaf3c84774c86efc7c09cd69237b8b45069b81f967e65cce392f328
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    seaborn==0.13.2 \
    spikeinterface[full,widgets]==0.103.0 \
    wavpack-numcodecs==0.2.2

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
