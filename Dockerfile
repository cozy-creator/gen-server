# Stage 1: Web-build stage
FROM node:22-bookworm-slim as builder

WORKDIR /app

# Copy the web folder
COPY web ./web

# Build the web folder
RUN cd ./web && \
    npm install && \
    npm run build


# Stage 2: production
FROM python:3.11.9-slim
# FROM nvidia/cuda:12.1.0-base-ubuntu22.04 as runtime

WORKDIR /app

# Configure apt-get to automatically use noninteractive settings
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# ENV PYTHONDONTWRITEBYTECODE=1

# Install Linux build and runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git curl build-essential libssl-dev libffi-dev wget ca-certificates \
    libgl1-mesa-glx libglib2.0-0 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch for CUDA 12.1
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio xformers

# Install Jupyter Lab
RUN pip install --no-cache-dir jupyterlab

# Install the latest version of diffusers straight from GitHub; the release branch may be out of date
RUN pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git

# Install the gen_server package and its plugin-packages
COPY packages/ ./packages
RUN pip install ./packages/gen_server[performance] && \
    pip install ./packages/image_utils && \
    pip install ./packages/core_extension_1

# start script
COPY scripts/start.sh .
RUN chmod +x start.sh
# remove any Windows line endings
RUN sed -i 's/\r$//' /app/start.sh

# Move files over from the web-build stage
COPY --from=builder /app/web/dist /srv/www/cozy/dist

# ENTRYPOINT ["./start.sh"]

CMD ["./start.sh"]
