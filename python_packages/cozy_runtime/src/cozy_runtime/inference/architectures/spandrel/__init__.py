from typing import Any, Optional
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

from cozy_runtime import StateDict
from cozy_runtime.base_types.architecture import (
    SpandrelArchitectureAdapter,
    ComponentMetadata,
)


_arch_instances = {
    "ESRGANArch": ESRGANArch(),
    "CRAFTArch": CRAFTArch(),
    "CompactArch": CompactArch(),
    "DnCNNArch": DnCNNArch(),
    "OmniSRArch": OmniSRArch(),
    "ATDArch": ATDArch(),
    "DATArch": DATArch(),
    "DCTLSAArch": DCTLSAArch(),
    "DITNArch": DITNArch(),
    "DRCTArch": DRCTArch(),
    "DRUNetArch": DRUNetArch(),
    "GRLArch": GRLArch(),
    "FBCNNArch": FBCNNArch(),
    "GFPGANArch": GFPGANArch(),
    "FFTformerArch": FFTformerArch(),
    "HATArch": HATArch(),
    "IPTArch": IPTArch(),
    "KBNetArch": KBNetArch(),
    "LaMaArch": LaMaArch(),
    "MixDehazeNetArch": MixDehazeNetArch(),
    "MMRealSRArch": MMRealSRArch(),
    "UformerArch": UformerArch(),
    "SwinIRArch": SwinIRArch(),
    "Swin2SRArch": Swin2SRArch(),
    "SPANArch": SPANArch(),
    "SwiftSRGANArch": SwiftSRGANArch(),
    "SCUNetArch": SCUNetArch(),
    "SAFMNBCIEArch": SAFMNBCIEArch(),
    "RGTArch": RGTArch(),
    "RestoreFormerArch": RestoreFormerArch(),
    "RealCUGANArch": RealCUGANArch(),
    "PLKSRArch": PLKSRArch(),
    "NAFNetArch": NAFNetArch(),
    "SAFMNArch": SAFMNArch(),
}


def detect_architecture(instance, state_dict: StateDict) -> Optional[ComponentMetadata]:
    return (
        ComponentMetadata(
            input_space=instance.id,
            output_space=instance.id,
            display_name=instance.name,
        )
        if instance.detect(state_dict)
        else None
    )


class ESRGANArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["ESRGANArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["ESRGANArch"], state_dict)


class CRAFTArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["CRAFTArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["CRAFTArch"], state_dict)


class CompactArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["CompactArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["CompactArch"], state_dict)


class DnCNNArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["DnCNNArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["DnCNNArch"], state_dict)


class OmniSRArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["OmniSRArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["OmniSRArch"], state_dict)


class ATDArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["ATDArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["ATDArch"], state_dict)


class DATArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["DATArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["DATArch"], state_dict)


class DCTLSAArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["DCTLSAArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["DCTLSAArch"], state_dict)


class DITNArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["DITNArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["DITNArch"], state_dict)


class DRCTArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["DRCTArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["DRCTArch"], state_dict)


class DRUNetArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["DRUNetArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["DRUNetArch"], state_dict)


class GRLArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["GRLArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["GRLArch"], state_dict)


class FBCNNArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["FBCNNArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["FBCNNArch"], state_dict)


class GFPGANArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["GFPGANArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["GFPGANArch"], state_dict)


class FFTformerArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["FFTformerArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["FFTformerArch"], state_dict)


class HATArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["HATArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["HATArch"], state_dict)


class IPTArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["IPTArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["IPTArch"], state_dict)


class KBNetArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["KBNetArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["KBNetArch"], state_dict)


class LaMaArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["LaMaArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["LaMaArch"], state_dict)


class MixDehazeNetArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["MixDehazeNetArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["MixDehazeNetArch"], state_dict)


class MMRealSRArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["MMRealSRArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["MMRealSRArch"], state_dict)


class UformerArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["UformerArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["UformerArch"], state_dict)


class SwinIRArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["SwinIRArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["SwinIRArch"], state_dict)


class Swin2SRArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["Swin2SRArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["Swin2SRArch"], state_dict)


class SPANArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["SPANArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["SPANArch"], state_dict)


class SwiftSRGANArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["SwiftSRGANArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["SwiftSRGANArch"], state_dict)


class SCUNetArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["SCUNetArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["SCUNetArch"], state_dict)


class SAFMNBCIEArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["SAFMNBCIEArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["SAFMNBCIEArch"], state_dict)


class RGTArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["RGTArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["RGTArch"], state_dict)


class RestoreFormerArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["RestoreFormerArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["RestoreFormerArch"], state_dict)


class RealCUGANArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["RealCUGANArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["RealCUGANArch"], state_dict)


class PLKSRArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["PLKSRArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["PLKSRArch"], state_dict)


class NAFNetArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["NAFNetArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["NAFNetArch"], state_dict)


class SAFMNArchitecture(SpandrelArchitectureAdapter):
    def __init__(self):
        super().__init__(_arch_instances["SAFMNArch"])

    @classmethod
    def detect(
        cls, state_dict: StateDict, **ignore: Any
    ) -> Optional[ComponentMetadata]:
        return detect_architecture(_arch_instances["SAFMNArch"], state_dict)


architectures = [
    ESRGANArchitecture,
    CRAFTArchitecture,
    CompactArchitecture,
    DnCNNArchitecture,
    OmniSRArchitecture,
    ATDArchitecture,
    DATArchitecture,
    DCTLSAArchitecture,
    DITNArchitecture,
    DRCTArchitecture,
    DRUNetArchitecture,
    GRLArchitecture,
    FBCNNArchitecture,
    GFPGANArchitecture,
    FFTformerArchitecture,
    HATArchitecture,
    IPTArchitecture,
    KBNetArchitecture,
    LaMaArchitecture,
    MixDehazeNetArchitecture,
    MMRealSRArchitecture,
    UformerArchitecture,
    SwinIRArchitecture,
    Swin2SRArchitecture,
    SPANArchitecture,
    SwiftSRGANArchitecture,
    SCUNetArchitecture,
    SAFMNBCIEArchitecture,
    RGTArchitecture,
    RestoreFormerArchitecture,
    RealCUGANArchitecture,
    PLKSRArchitecture,
    NAFNetArchitecture,
    SAFMNArchitecture,
]
