from typing import Optional, Dict, Any
from cozy_runtime.base_types import CustomNode


class LoadLoraNode(CustomNode):
    """Custom node to load a Lora model."""

    def __init__(self):
        super().__init__()

    def __call__(  # type: ignore
        self,
        lora_path: str,
        model_scale: float = 1.0,
        text_encoder_scale: float = 1.0,
        text_encoder_2_scale: float = 1.0,
        adapter_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load a Lora model.

        Args:
            lora_path: The path to the Lora model in the format 'repo_id/model_name/weight_name'.
            model_scale: The scale to apply to the model (unet or transformer) weights.
            text_encoder_scale: The scale to apply to the text encoder weights.
            text_encoder_2_scale: The scale to apply to the second text encoder weights.
            adapter_name: The name of the adapter to use. If not provided, the model name will be used.

        Returns:
            A dictionary containing the repo_id, weight_name, adapter_name, model_scale, text_encoder_scale, and text_encoder_2_scale.
        """

        # Split the lora_path into repo_id and weight_name
        parts = lora_path.split("/", 2)
        if len(parts) < 3:
            raise ValueError(
                "Invalid lora_path format. Expected format: 'repo_id/model_name/weight_name'"
            )

        repo_id = f"{parts[0]}/{parts[1]}"
        weight_name = parts[2]

        # If adapter_name is not provided, use the model name as the adapter name
        if adapter_name is None:
            adapter_name = "test"

        return {
            "repo_id": repo_id,
            "weight_name": weight_name,
            "adapter_name": adapter_name,
            "model_scale": model_scale,
            "text_encoder_scale": text_encoder_scale,
            "text_encoder_2_scale": text_encoder_2_scale,
        }
