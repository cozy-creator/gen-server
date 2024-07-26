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

# Move files over from the web-build stage
COPY --from=builder /app/web/dist /srv/www/cozy/dist

COPY packages/ ./packages
COPY pyproject.toml ./pyproject.toml

# Install the latest unreleased version of Diffusers
RUN pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git

# Install the gen_server package and its plugin-packages
RUN pip install --no-cache-dir --prefer-binary ./packages/gen_server[performance] && \
    pip install ./packages/image_utils && \
    pip install ./packages/core_extension_1

# Install Jupyter Lab
RUN pip install --no-cache-dir jupyterlab
# RUN mkdir -p /root/.local/share/jupyter/runtime && \
#     chmod 777 /root/.local/share/jupyter/runtime

# start script
COPY scripts/start.sh .
# remove any Windows line endings
RUN sed -i 's/\r$//' /app/start.sh
RUN chmod +x start.sh

# ENTRYPOINT ["./start.sh"]

CMD ["./start.sh"]
