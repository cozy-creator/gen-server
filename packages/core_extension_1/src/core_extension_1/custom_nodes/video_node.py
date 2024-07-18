import torch
from gen_server.base_types import CustomNode
import torchvision.io as tvio
import os
import json

class VideoNode(CustomNode):
    """Loads a video from a file path or URL."""

    def __call__(self, file_path: str) -> dict[str, torch.Tensor]:
        """
        Args:
            file_path: Path or URL of the video file.
        Returns:
            A dictionary containing the loaded video as a tensor.
        """
        try:
            video_tensor = tvio.read_video(file_path, pts_unit='sec')[0] 
            return {"video": video_tensor}
        except Exception as e:
            raise ValueError(f"Error loading video: {e}")

    @staticmethod
    def get_spec():
        """Returns the node specification."""
        spec_file = os.path.join(os.path.dirname(__file__), 'video_node.json')
        with open(spec_file, 'r', encoding='utf-8') as f:
            spec = json.load(f)
        return spec