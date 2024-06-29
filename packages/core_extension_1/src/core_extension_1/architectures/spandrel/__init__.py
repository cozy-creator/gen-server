from typing import Any, Optional, Type

from spandrel import Architecture as SpandrelArchitecture
from spandrel.architectures.ESRGAN import ESRGANArch
from spandrel.architectures.CRAFT import CRAFTArch
from spandrel.architectures.Compact import CompactArch
from spandrel.architectures.DnCNN import DnCNNArch
from spandrel.architectures.OmniSR import OmniSRArch
from spandrel.architectures.ATD import ATDArch
from spandrel.architectures.DAT import DATArch
from spandrel.architectures.DCTLSA import DCTLSAArch
from spandrel.architectures.DITN import DITNArch
from spandrel.architectures.DRCT import DRCTArch
from spandrel.architectures.DRUNet import DRUNetArch
from spandrel.architectures.GRL import GRLArch
from spandrel.architectures.FBCNN import FBCNNArch
from spandrel.architectures.GFPGAN import GFPGANArch
from spandrel.architectures.FFTformer import FFTformerArch
from spandrel.architectures.HAT import HATArch
from spandrel.architectures.IPT import IPTArch
from spandrel.architectures.KBNet import KBNetArch
from spandrel.architectures.LaMa import LaMaArch
from spandrel.architectures.MixDehazeNet import MixDehazeNetArch
from spandrel.architectures.MMRealSR import MMRealSRArch
from spandrel.architectures.Uformer import UformerArch
from spandrel.architectures.SwinIR import SwinIRArch
from spandrel.architectures.Swin2SR import Swin2SRArch
from spandrel.architectures.SPAN import SPANArch
from spandrel.architectures.SwiftSRGAN import SwiftSRGANArch
from spandrel.architectures.SCUNet import SCUNetArch
from spandrel.architectures.SAFMNBCIE import SAFMNBCIEArch
from spandrel.architectures.RGT import RGTArch
from spandrel.architectures.RestoreFormer import RestoreFormerArch
from spandrel.architectures.RealCUGAN import RealCUGANArch
from spandrel.architectures.PLKSR import PLKSRArch
from spandrel.architectures.NAFNet import NAFNetArch
from spandrel.architectures.SAFMN import SAFMNArch

from gen_server import StateDict
from gen_server.base_types.architecture import (
    SpandrelArchitectureAdapter,
    ComponentMetadata,
)

architectures = []
_spandrel_architectures = [
    ESRGANArch,
    CRAFTArch,
    CompactArch,
    DnCNNArch,
    OmniSRArch,
    ATDArch,
    DATArch,
    DCTLSAArch,
    DITNArch,
    DRCTArch,
    DRUNetArch,
    FBCNNArch,
    GFPGANArch,
    FFTformerArch,
    HATArch,
    IPTArch,
    KBNetArch,
    LaMaArch,
    MixDehazeNetArch,
    MMRealSRArch,
    UformerArch,
    SwinIRArch,
    Swin2SRArch,
    SPANArch,
    SwiftSRGANArch,
    SCUNetArch,
    SAFMNBCIEArch,
    RGTArch,
    RestoreFormerArch,
    RealCUGANArch,
    PLKSRArch,
    NAFNetArch,
    SAFMNArch,
    GFPGANArch,
    FBCNNArch,
    GRLArch,
]


# def build_architecture(architecture):
#     arch_instance = architecture()
#     if not isinstance(arch_instance, SpandrelArchitecture):
#         raise ValueError(
#             f"Architecture must be an instance of SpandrelArchitecture, got {architecture}"
#         )
#
#     def __init__(self):
#         super(self.__class__, self).__init__(arch_instance)
#
#     @classmethod
#     def detect(
#         cls,
#         state_dict: StateDict = None,
#         metadata: dict[str, Any] = None,
#     ) -> Optional[ComponentMetadata]:
#         return (
#             ComponentMetadata(
#                 display_name=arch_instance.name,
#                 input_space=arch_instance.id,
#                 output_space=arch_instance.id,
#             )
#             if arch_instance.detect(state_dict)
#             else None
#         )
#
#     cls_attributes = {"__init__": __init__, "detect": detect}
#     return type(architecture.__name__, (SpandrelArchitectureAdapter,), cls_attributes)
#


def build_architecture(
    architecture: Type[SpandrelArchitecture],
) -> Type[SpandrelArchitectureAdapter]:
    arch_instance = architecture()
    if not isinstance(arch_instance, SpandrelArchitecture):
        raise ValueError(
            f"Architecture must be an instance of SpandrelArchitecture, got {type(architecture).__name__}"
        )

    class DynamicArchitecture(SpandrelArchitectureAdapter):
        def __init__(self):
            super().__init__(arch_instance)

        @classmethod
        def detect(
            cls,
            state_dict: Optional[StateDict] = None,
            metadata: Optional[dict[str, Any]] = None,
        ) -> Optional[ComponentMetadata]:
            if arch_instance.detect(state_dict):
                return ComponentMetadata(
                    display_name=arch_instance.name,
                    input_space=arch_instance.id,
                    output_space=arch_instance.id,
                )
            return None

    DynamicArchitecture.__name__ = architecture.__name__
    DynamicArchitecture.__qualname__ = architecture.__qualname__
    return DynamicArchitecture


for _architecture in _spandrel_architectures:
    architectures.append(build_architecture(_architecture))
