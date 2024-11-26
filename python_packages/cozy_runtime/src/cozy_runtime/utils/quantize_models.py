from optimum.quanto.models import QuantizedTransformersModel, QuantizedDiffusersModel
from transformers import T5EncoderModel
from diffusers import FluxTransformer2DModel
from optimum.quanto import freeze, qfloat8, quantize
import torch


class QuantizedT5EncoderModelForCausalLM(QuantizedTransformersModel):
    auto_class = T5EncoderModel


class QuantizedFluxTransformer2DModel(QuantizedDiffusersModel):
    base_class = FluxTransformer2DModel


def quantize_model_fp8(model: torch.nn.Module):
    quantize(model, weights=qfloat8)
    freeze(model)


