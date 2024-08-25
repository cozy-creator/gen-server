from gen_server.base_types import CustomNode
from PIL import Image
import torch
import numpy as np
from safetensors import torch as sf
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0

class CompositeImagesNode(CustomNode):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.unet = self._load_unet()
        self.vae = self._load_vae()
        self._load_iclight_model()

    def _load_unet(self):
        unet = UNet2DConditionModel.from_pretrained('stablediffusionapi/realistic-vision-v51', subfolder="unet")
        
        with torch.no_grad():
            new_conv_in = torch.nn.Conv2d(8, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
            new_conv_in.bias = unet.conv_in.bias
            unet.conv_in = new_conv_in

        unet_original_forward = unet.forward
        def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs): # type: ignore
            c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
            c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
            new_sample = torch.cat([sample, c_concat], dim=1)
            kwargs['cross_attention_kwargs'] = {}
            return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)
        unet.forward = hooked_unet_forward

        unet = unet.to(device=self.device, dtype=torch.float16)
        unet.set_attn_processor(AttnProcessor2_0())
        return unet

    def _load_vae(self):
        vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="vae")
        vae = vae.to(device=self.device, dtype=torch.bfloat16)
        vae.set_attn_processor(AttnProcessor2_0())
        return vae

    def _load_iclight_model(self):
        model_path = './models/iclight_sd15_fc.safetensors'
        sd_offset = sf.load_file(model_path)
        sd_origin = self.unet.state_dict()
        sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
        self.unet.load_state_dict(sd_merged, strict=True)

    @torch.inference_mode()
    async def __call__(self, foreground: Image.Image, background: Image.Image) -> dict[str, Image.Image]: # type: ignore
        background = background.resize(foreground.size, Image.LANCZOS)
        
        fg_tensor = self._numpy2pytorch([np.array(foreground)])
        bg_tensor = self._numpy2pytorch([np.array(background)])

        fg_latent = self.vae.encode(fg_tensor.to(self.vae.dtype)).latent_dist.mode() * self.vae.config.scaling_factor
        bg_latent = self.vae.encode(bg_tensor.to(self.vae.dtype)).latent_dist.mode() * self.vae.config.scaling_factor

        composited_latent = self.unet(
            bg_latent,
            torch.zeros(1).to(self.device),
            encoder_hidden_states=torch.zeros(1, 77, 768).to(self.device),
            cross_attention_kwargs={'concat_conds': fg_latent}
        ).sample

        composited_image = self.vae.decode(composited_latent / self.vae.config.scaling_factor).sample
        composited_image = self._pytorch2numpy(composited_image)[0]

        return {"composited_image": Image.fromarray(composited_image)}

    def _numpy2pytorch(self, imgs): # type: ignore
        h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0
        h = h.permute(0, 3, 1, 2).to(self.device)
        return h

    def _pytorch2numpy(self, imgs): # type: ignore
        imgs = imgs.permute(0, 2, 3, 1)
        imgs = (imgs * 127.5 + 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
        return imgs
    