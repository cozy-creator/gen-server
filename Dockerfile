FROM python:3.11.9-slim

WORKDIR /app

COPY packages/ ./packages
COPY pyproject.toml ./pyproject.toml

# Install build dependencies without asking for confirmation
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    wget \
    ca-certificates \
    && apt-get clean

RUN rm -rf /var/lib/apt/lists/*

# Install Rust and Cargo
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install gen_server dependencies
WORKDIR /app/packages/gen_server
RUN pip install -e .

# Install core_extension_1 dependencies
WORKDIR /app/packages/core_extension_1
RUN pip install -e .

# Install image_utils dependencies
WORKDIR /app/packages/image_utils
RUN pip install -e .

# Back to the app directory
WORKDIR /app

CMD ["comfy-creator"]
