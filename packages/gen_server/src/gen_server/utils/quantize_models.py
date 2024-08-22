from optimum.quanto.models import QuantizedTransformersModel, QuantizedDiffusersModel
from transformers import T5EncoderModel
from diffusers import FluxTransformer2DModel




class QuantizedT5EncoderModelForCausalLM(QuantizedTransformersModel):
    auto_class = T5EncoderModel


class QuantizedFluxTransformer2DModel(QuantizedDiffusersModel):
    base_class = FluxTransformer2DModel

