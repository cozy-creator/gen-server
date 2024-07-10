# Stage 1: Build stage
FROM python:3.11.9-slim as builder

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
    pip install ./packages/core_extension_1 && \
    pip install ./packages/image_utils

# Stage 2: Final stage
FROM python:3.11.9-slim

WORKDIR /app

# Copy only the necessary files from the build stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY pyproject.toml ./pyproject.toml

CMD ["cozy", "run"]
