from gen_server.base_types.architecture import Architecture
from packages.core_extension_1.src.core_extension_1.architectures.esrgan_archs.real_esrgan_x2 import (
    RealESRGAN,
)

if __name__ == "__main__":
    arch = RealESRGAN()
    #
    # try:
    #     print(verify.verifyObject(IArchitecture, testt))
    # except Invalid as e:
    #     logging.log(logging.WARNING, f"Error in verifying component IArchitecture {e}")
    # except Exception:
    #     logging.log("Unknownerror...")
    # print(IArchitecture.providedBy(testt))
    print(isinstance(arch, Architecture))

    print(arch.display_name)
    print(arch.input_space)
    print(arch.output_space)
