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

from gen_server.base_types.architecture import SpandrelArchitectureAdapter


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


def build_architecture(architecture):
    if not issubclass(architecture, SpandrelArchitecture):
        raise ValueError(
            f"Architecture must be a subclass of SpandrelArchitecture, got {architecture}"
        )

    def __init__(self):
        super(self.__class__, self).__init__(architecture())

    cls_attributes = {"__init__": __init__}
    return type(architecture.__name__, (SpandrelArchitectureAdapter,), cls_attributes)


for _architecture in _spandrel_architectures:
    architectures.append(build_architecture(_architecture))
