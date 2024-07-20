# Stage 1: Web-build stage
FROM node:22-bookworm-slim as builder

WORKDIR /app

# Copy the web folder
COPY web ./web

# Build the web folder
RUN cd ./web && \
    npm install && \
    npm run build


# Stage 2: Python-build stage
FROM python:3.11.9-slim as python-builder

WORKDIR /app

# Configure apt-get to automatically use noninteractive settings
ENV DEBIAN_FRONTEND=noninteractive

COPY packages/ ./packages
COPY pyproject.toml ./pyproject.toml

# Install build dependencies without asking for confirmation
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git curl build-essential libssl-dev libffi-dev wget ca-certificates \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install the gen_server package and its plugin-packages
RUN pip install --no-cache-dir --prefer-binary ./packages/gen_server && \
    pip install ./packages/image_utils && \
    pip install ./packages/core_extension_1


# Stage 3: Final stage
FROM python:3.11.9-slim

WORKDIR /app

# Copy only the necessary files from the build stages
COPY --from=python-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/web/dist /srv/www/cozy/dist

COPY pyproject.toml ./pyproject.toml

# Install PyTorch for CUDA 12.1
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

# Install Linux runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*


CMD ["cozy", "run"]
