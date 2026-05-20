# hash:sha256:c72f63431637e3f573a7f288ef979e5fe4d60bc11564a9ca50d560dd0baa8a8d
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN pip install -U --no-cache-dir \
    s3fs \
    seaborn==0.13.2 \
    spikeinterface[full,widgets]==0.103.0 \
    wavpack-numcodecs==0.2.2
