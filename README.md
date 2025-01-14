### Cozy Creator - Gen Server

> **Note:** This repo is not yet useable or functional, and contains a lot of dead code. I'll clean it up and remove this notice later when we have our first actual release.

This is an attempt to create a better version of ComfyUI, using standard libraries. (React Flow for the UX, and Diffusers for the generation.) Originally this started off as a Node.js wrapper around ComfyUI, and then I discarded ComfyUI and rebuilt it in Python, and then I discarded that and rebuilt it in Golang.

Golang is a vastly superior language to Python (10x CPU compute time, lower latency, native concurrency), but it doesn't have support for Diffusers, PyTorch, or most AI libraries. So right now the plan is to run Golang and Python processes in parallel, and have Golang <-> Python communication using TCP. Golang handles the network while Python handles the GPU.

If anyone has any ideas on how to skip Python and do just Golang <-> C / CUDA that would be preferrable. This is my first time creating a Golang and Python project so everything may be a bit rough here.

If you have suggestions or would like to work together on this, open an issue / PR or email me directly at [paul@fidika.com](mailto:paul@fidika.com)


### Other Notes

This repo is the backend used to power [cozy.art](https://cozy.art) using [Runpod GPUs](https://runpod.io) currently.

Documentation will be here: [Docs](https://cozy.art/docs)

The graph editor front-end is here: [https://github.com/cozy-creator/graph-editor](https://github.com/cozy-creator/graph-editor)

The graph editor is currently not yet functional, but will be functional... eventually.

I think it's best to focus on the backend first and then rebuild the graph-editor after the server is stable; trying to build both at once was challenging. I originally built and planned an entire plugin system, like how ComfyUI has custom nodes--but building a plugin system correctly is very hard so that's on hold for now--it'll come eventually.

The intention is for this to be runnable locally AND in the cloud. Right now I'm focusing on the cloud-aspect, since that's most relevant to cozy-art, but eventually it will run locally on consumer hardware as well. Running locally is important because I don't want to have to censor or monitor how consumers use this software; they should be able to use it however they want.


### License

The plan is to release this all under a non-commercial usage license, and charge a licensing fee for those who want to use it commercially. I think this is a good balance between making the source-code available, allowing consumers to run locally, allowing devs to host their own instances (and not just rely on an API!), and yet still having a profitable business model that will make development sustainable.


### Configuration and Environment Variables

(TO DO)

Note that relative file paths are relative to the current working directory, not the location of the gen-server executable. You can use relative file paths with flags like `--public-dir` and `--env-file`.

It's recommended to use the config.yaml file as a config-map if you're deploying this in a Kubernetes environment.


### Building Docker Container

Single architecture build (will build for your system's architecture):
```sh
docker build -t cozycreator/gen-server:0.4.2 .
```

> **Note:** I'm unable to get QEMU to work, so that I can compile to arm64 from my x86 machine. Instead, I use a GitHub action to start an arm64 runner to build the container instead. If you pull the container from dockerhub, you should get the x86 or arm64 version that matches your machine.

### Running Container Locally

Run locally on your Windows machine:
```sh
docker run -d \
    --name gen-server \
    --gpus all \
    -v ~/.cozy-creator:/root/.cozy-creator \
    --env-file .env.local \
    -p 8881:8881 \
    -p 8888:8888 \
    cozycreator/gen-server:0.4.1
```

### Running Container on Runpod

(TO DO)


### Docker Container Notes:

This container only works on linux + amd64 machines (which includes Windows because Windows runs docker containers inside of WSL2 (Windows Subsystem for Linux)).

Note that I wanted to create a multi-platform build, but everything else is useless because:

(1) when running docker containers on MacOS (Darwin) the container runs inside of Linux, which does not have access to MPS (Apple's version of CUDA)

(2) when running on a Linux/arm64 machine, pytorch installs itself in CPU mode because pytorch does not have any pre-built wheels for arm64 + CUDA. (I believe???) Note also that xformers also doesn't support ARM64 at all.


RUN if [ "$(uname -m)" = "aarch64" ]; then \
        # On ARM64 host, install AMD64 cross-compilation tools
        dpkg --add-architecture amd64 && \
        apt-get update && \
        apt-get install -y --no-install-recommends \
        gcc-x86-64-linux-gnu libc6-dev-amd64-cross \
        libsqlite3-dev:amd64 libsqlite3-dev ca-certificates; \
    elif [ "$(uname -m)" = "x86_64" ]; then \
        # On AMD64 host, install ARM64 cross-compilation tools

# Build the Go binary
RUN echo "Building for architecture: $GOARCH and OS: $GOOS" && \
    if [ "$GOARCH" = "amd64" ] && [ "$(uname -m)" = "aarch64" ]; then \
        # Cross-compile from arm64 to amd64
        CC=x86_64-linux-gnu-gcc CGO_ENABLED=1 GOOS=$GOOS GOARCH=$GOARCH go build -o cozy .; \
    elif [ "$GOARCH" = "arm64" ] && [ "$(uname -m)" = "x86_64" ]; then \
        # Cross-compile from amd64 to arm64
        CC=aarch64-linux-gnu-gcc CGO_ENABLED=1 GOOS=$GOOS GOARCH=$GOARCH go build -o cozy .; \

# FROM python:3.11.9-slim
# FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04 AS runtime
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime


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