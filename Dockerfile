# These ARGs can be overridden during build
ARG TARGETARCH
ARG TARGETOS=linux

# Stage 1: Build the web bundle
FROM node:22-bookworm-slim AS web-builder

WORKDIR /app

# Copy the web folder
COPY web ./web

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget unzip ca-certificates && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# temporarily download the dist folder from the web-builder stage as the build is currently broken
RUN wget https://github.com/user-attachments/files/17084728/dist.zip -O /app/web/dist.zip && \
    unzip /app/web/dist.zip -d /app/web/dist

# Build the web folder
# RUN cd ./web && \
#     npm install && \
#     npm run build


# Stage 2: Build the Go binary
FROM golang:1.23 AS go-builder

# Import the build arguments
ARG TARGETARCH
ARG TARGETOS

WORKDIR /src

# Copy go mod and sum files
COPY go.mod go.sum ./

# Download all dependencies
RUN go mod download

# Copy only the the Golang source code needed for the build
# COPY cmd/ internal/ pkg/ tools/ .
COPY cmd cmd
COPY internal internal
COPY pkg pkg
COPY scripts scripts
COPY tools tools
COPY main.go .

# Build the Go binary
RUN CGO_ENABLED=1 GOOS=$TARGETOS GOARCH=$TARGETARCH go build -o cozy .


# Stage 3: Build Python environment and final image
# FROM python:3.11.9-slim
# FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04 AS runtime
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime

# Import the build arguments again for the final stage
ARG TARGETARCH
ARG TARGETOS

WORKDIR /app

# Configure apt-get to automatically use noninteractive settings
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# ENV PYTHONDONTWRITEBYTECODE=1

# Install Linux build and runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git curl build-essential libssl-dev libffi-dev wget unzip nano ca-certificates software-properties-common \
    libgl1 libglx-mesa0 libglib2.0-0 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install CUDA toolkit 12.6, conditional on architecture
# RUN if [ "$GOARCH" = "amd64" ]; then \
#       wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
#       dpkg -i cuda-keyring_1.1-1_all.deb && add-apt-repository contrib && \
#       apt-get update && apt-get install -y cuda-toolkit-12-6; \
#     elif [ "$GOARCH" = "arm64" ]; then \
#       echo "CUDA for ARM64 is not directly available from this repository. Please check NVIDIA's ARM support."; \
#     else \
#       echo "Unsupported architecture for CUDA installation"; \
#     fi

# Install Python 3.11; use deadsnakes PPA to get the latest version
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y python3.11 python3.11-dev python3-pip python3.11-venv && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment, otherwise we get a PEP 668 error
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch for CUDA 12.4 if possible
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        pip install -U --no-cache-dir torch torchvision torchaudio \
        xformers --index-url https://download.pytorch.org/whl/cu124; \
    else \
        echo "xformers is unavailable on $TARGETARCH architecture"; \
        pip3 install torch torchvision torchaudio; \
    fi

# Install Jupyter Lab
RUN pip install --no-cache-dir jupyterlab

# Install the latest version of diffusers straight from GitHub
# Hugging face's official release may be out of date
RUN pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git

# Install the cozy_runtime python package
COPY python_packages/ ./python_packages
RUN pip install ./python_packages/cozy_runtime[performance]

# Copy the web bundle we built in stage-1
COPY --from=web-builder /app/web/dist /srv/www/cozy
COPY --from=go-builder /src/cozy /usr/local/bin/cozy

# Copy start script
COPY scripts/start.sh .

# remove any Windows line endings, just in case
RUN chmod +x start.sh && \
    sed -i 's/\r$//' /app/start.sh

# Gen-server and Jupyter Lab ports
EXPOSE 8881 8888

CMD ["./start.sh"]
