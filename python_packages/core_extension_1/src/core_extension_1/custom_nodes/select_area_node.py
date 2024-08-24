import torch
from gen_server.base_types import CustomNode
from segment_anything import sam_model_registry, SamPredictor
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops
from PIL import Image
import numpy as np
import scipy.ndimage
import os
import json

class SelectAreaNode(CustomNode):
    """Selects an area in an image based on a text prompt using GroundingDino and SAM."""

    def __init__(self):
        super().__init__()
        self.sam_predictor = self.load_sam()
        self.grounding_dino_model = self.load_groundingdino()

    def __call__(self, image: torch.Tensor, text_prompt: str, feather_radius: int = 0) -> dict[str, Image.Image]:
        """
        Args:
            image: Input image tensor (C, H, W) or PIL Image.
            text_prompt: Text prompt describing the area to be selected.
        Returns:
            A dictionary containing the mask of the selected area as a PIL Image.
        """
        try:
            if isinstance(image, torch.Tensor):
                image = Image.fromarray((image * 255).permute(1, 2, 0).numpy().astype(np.uint8))
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                raise TypeError("Input image must be a torch.Tensor, np.ndarray, or PIL Image.")

            image_np = np.array(image)
            self.sam_predictor.set_image(image_np)

            # Detect objects using GroundingDINO
            boxes, _, _ = self.detect_objects(image, text_prompt)

            if len(boxes) > 0:
                combined_mask = np.zeros(image_np.shape[:2], dtype=bool)
                for box in boxes:
                    # Generate mask using SAM
                    input_box = box.cpu().numpy()
                    masks, _, _ = self.sam_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None, :],
                        multimask_output=False,
                    )
                    combined_mask = np.logical_or(combined_mask, masks[0])

                # Feather the mask (optional)
                combined_mask = self.feather_mask(combined_mask, iterations=feather_radius)  # Adjust iterations as needed

                mask_image = Image.fromarray(combined_mask.astype(np.uint8) * 255)
                return {"mask": mask_image}
            else:
                raise ValueError(f"No objects matching '{text_prompt}' found in the image.")

        except Exception as e:
            raise ValueError(f"Error selecting area: {e}")

    def load_sam(self) -> SamPredictor:
        sam_checkpoint = "models/sam_vit_h_4b8939.pth"  # Update path if necessary
        model_type = "vit_h"
        device = self.DEVICE
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        return SamPredictor(sam)

    def load_groundingdino(self) -> torch.nn.Module:
        config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"  # Update path
        checkpoint_file = "models/groundingdino_swint_ogc.pth"  # Update path
        model = load_model(config_file, checkpoint_file)
        model.to(self.DEVICE)
        return model

    def transform_image(self, image: Image.Image) -> torch.Tensor:
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_transformed, _ = transform(image, None)
        return image_transformed

    def detect_objects(self, image: Image.Image, text_prompt: str) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        image_transformed = self.transform_image(image)
        boxes, logits, phrases = predict(
            model=self.grounding_dino_model,
            image=image_transformed,
            caption=text_prompt,
            box_threshold=0.3,  # Adjust thresholds as needed
            text_threshold=0.25,
        )
        W, H = image.size
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([W, H, W, H])
        return boxes, logits, phrases

    def feather_mask(self, mask: np.ndarray, iterations: int = 5) -> np.ndarray:
        """Feather the mask to create a soft transition (optional)."""
        mask = mask.astype(np.float32)
        for _ in range(iterations):
            mask = scipy.ndimage.gaussian_filter(mask, sigma=1)
            mask[mask > 0] = 1
        return mask

    @staticmethod
    def get_spec():
        """Returns the node specification."""
        spec_file = os.path.join(os.path.dirname(__file__), 'select_area_node.json')
        with open(spec_file, 'r', encoding='utf-8') as f:
            spec = json.load(f)
        return spec