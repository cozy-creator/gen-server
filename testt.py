from gen_server import Architecture
from packages.core_extension_1.src.core_extension_1.architectures.spandrel import (
    architectures,
)

if __name__ == "__main__":
    for arch in architectures:
        print(issubclass(arch, Architecture))

        a = arch()
        print(isinstance(a, Architecture))

        # print(a.input_space)
        # print(arch.input_space)
        # print(arch.output_space)
