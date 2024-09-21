# Stage 1: Build the web bundle
FROM node:22-bookworm-slim as web-builder

WORKDIR /app

# Copy the web folder
COPY web ./web

# temporarily download the dist folder from the web-builder stage as the build is currently broken
RUN wget https://github.com/user-attachments/files/17084728/dist.zip -O /app/web/dist.zip \
    && unzip /app/web/dist.zip -d /app/web/dist

# Build the web folder
# RUN cd ./web && \
#     npm install && \
#     npm run build


# Stage 2: Build the Go binary
FROM golang:1.23 AS go-builder

WORKDIR /app

# Copy go mod and sum files
COPY go.mod go.sum ./

# Download all dependencies
RUN go mod download

# Copy only the the Golang source code needed for the build
# COPY cmd/ internal/ pkg/ tools/ .
COPY cmd cmd
COPY pkg pkg
COPY tools tools
COPY internal internal
COPY main.go .

# Build the Go binary
ARG GOARCH=amd64
RUN CGO_ENABLED=0 GOOS=linux GOARCH=$GOARCH go build -o cozy-server .


# Stage 3: Build Python environment and final image
FROM python:3.11.9-slim
# FROM nvidia/cuda:12.4.0-base-ubuntu22.04 as runtime

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

# Install PyTorch for CUDA 12.4
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
    torch torchvision torchaudio xformers

# Install Jupyter Lab
RUN pip install --no-cache-dir jupyterlab

# Install the latest version of diffusers straight from GitHub
# Hugging face's official release may be out of date
RUN pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git

# Install the gen_server package and its plugin-python_packages
COPY python_packages/ ./python_packages
RUN pip install ./python_packages/gen_server[performance] && \
    pip install ./python_packages/image_utils && \
    pip install ./python_packages/core_extension_1

# Copy the web bundle we built in stage-1
COPY --from=web-builder ./web/dist /srv/www/cozy/dist

# Copy the binary we built in stage-2
COPY --from=go-builder /app/cozy-server /usr/local/bin/cozy-server

# Copy start script
COPY scripts/start.sh .
RUN chmod +x start.sh
# remove any Windows line endings, just in case
RUN sed -i 's/\r$//' /app/start.sh

EXPOSE 9009 9008

CMD ["./start.sh"]