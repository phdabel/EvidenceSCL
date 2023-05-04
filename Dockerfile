# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10-alpine3.17

ARG UID=1000

# Install dependencies
RUN apk add --no-cache --virtual .build-deps \
    bash \
    git \
    gfortran \
    openssh-client \
    curl \
    jq \
    zip \
    unzip \
    rsync \
    openssh \
    openssl \
    ca-certificates \
    groff \
    less \
    gcc \
    g++ \
    libc-dev \
    make \
    cmake \
    libxslt-dev \
    libxml2-dev \
    libffi-dev \
    openblas \
    openblas-dev \
    openssl \
    openssl-dev \
    python3-dev \
    musl-dev \
    libgcc \
    libstdc++ \
    binutils \
    libmagic \
    libpq \
    cargo

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Update pip
RUN pip install --upgrade pip

WORKDIR /app

# Creates a non-root user and adds permission to access the /app folder
RUN adduser -u ${UID} --disabled-password --gecos "" appuser && \
    chown -R appuser /app

USER appuser

# Install torch 2.0 CPU
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
