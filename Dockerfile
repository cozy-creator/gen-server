FROM python:3.11.9-slim
# FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04 as runtime

WORKDIR /app

# Configure apt-get to automatically use noninteractive settings
ENV DEBIAN_FRONTEND=noninteractive

COPY packages/ ./packages
COPY pyproject.toml ./pyproject.toml

# Install build dependencies without asking for confirmation
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git curl build-essential libssl-dev libffi-dev wget ca-certificates \
    && apt-get clean

RUN rm -rf /var/lib/apt/lists/*

# Install Rust and Cargo
# RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
# ENV PATH="/root/.cargo/bin:${PATH}"

# Install the gen_server package and its plugin-packages
RUN pip install --no-cache-dir --prefer-binary -e ./packages/gen_server && \
    pip install -e ./packages/core_extension_1 && \
    pip install -e ./packages/image_utils

CMD ["comfy-creator"]
