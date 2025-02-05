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

To build a container for your system's native CPU architecture, execute the following in this folder:
```sh
docker build -t cozycreator/gen-server:local .
```

Pull a prebuilt docker image from our dockerhub:

```sh
docker pull cozycreator/gen-server:latest
```

> **Note:** We use GitHub Actions to auto-build docker images. ':latest' refers to the current head of the master branch, while tags like ':0.4.2' refers to tagged releases. We have images that support both x86 and arm64 architectures, although x86 CPUS are better supported.

### Running Container Locally

Create a .env.local file with your desired configuration (copy and modify the .env.local.example file). This command will mount your local .cozy-creator folder, where models are stored, into the docker container, so it'll have access to all models downloaded locally.
```sh
docker run -d \
    --name gen-server \
    --gpus all \
    --rm \
    -v ${HOME}/.cozy-creator:/root/.cozy-creator \
    --env-file .env.local \
    -p 8881:8881 \
    -p 8888:8888 \
    cozycreator/gen-server:latest
```

View the status of all your docker containers:
```sh
docker ps -a
```

Enter the container:
```sh
docker exec -it gen-server /bin/bash
```

View the logs of your docker container:
```sh
docker logs -f gen-server
```

To view the graph-editor GUI, go to `http://localhost:8881/`
To view the Jupyter Lab GUI, go to `http://localhost:8888/`

### Running Container on Runpod

(TO DO)


### Docker Containers on MacOS:

Note that Docker containers are not very useful on MacOS (Darwin) because the container runs inside of Linux, which does not have access to the system's GPU (MPS, Apple's version of CUDA). It's recommended you install the gen-server binary and cozy-runtime directly on your MacOS machine.
