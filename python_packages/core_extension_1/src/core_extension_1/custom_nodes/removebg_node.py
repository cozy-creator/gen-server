from gen_server.base_types import CustomNode
from PIL import Image
import torch
import torchvision.transforms as T
from typing import Union, Dict
from gen_server.globals import get_architectures
from gen_server.utils.paths import get_models_dir
from gen_server.utils.load_models import load_state_dict_from_file
import os

class RemoveBackgroundNode(CustomNode):
    def __init__(self):
        super().__init__()
        self.model = self._load_model()

    def _load_model(self):
        architectures = get_architectures()
        print(architectures)
        BriaRMBG = architectures["core_extension_1.briarmbg"]
        
        device = self._get_device()
        models_dir = get_models_dir()
        
        rmbg = BriaRMBG()
        rmbg.load(
            load_state_dict_from_file(
                os.path.join(models_dir, "model.pth"), device=device
            )
        )
        
        return rmbg

    def _get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    async def __call__(self, image: Union[Image.Image, torch.Tensor]) -> Dict[str, Image.Image]: # type: ignore
        device = self._get_device()

        print("RemoveBackgroundNode: __call__")
        
        if isinstance(image, Image.Image):
            original_image = image
            image = T.ToTensor()(image).unsqueeze(0)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            original_image = T.ToPILImage()(image.squeeze(0))
        
        # image = image.to(device)

        print("RemoveBackgroundNode: before torch.no_grad()")
        
        with torch.no_grad():
            print("RemoveBackgroundNode: inside torch.no_grad()")
            output = self.model.model(image)
            if isinstance(output, list):
                mask = output[0][0]
            elif isinstance(output, tuple):
                mask = output[0][0]
            else:
                mask = output

        # Convert mask to PIL Image
        mask_pil = T.ToPILImage()(mask.squeeze(0))
        
        # Apply the mask to the original image
        image_rgba = original_image.convert("RGBA")
        mask_rgba = mask_pil.convert("L").resize(image_rgba.size)
        background = Image.new("RGBA", image_rgba.size, (0, 0, 0, 0))
        foreground = Image.composite(image_rgba, background, mask_rgba)
        
        return {"foreground": foreground}
