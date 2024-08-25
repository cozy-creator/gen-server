import torch
import torchaudio
from gen_server.base_types import CustomNode
import os
import json
from typing import Union

class AudioNode(CustomNode):
    """Loads an audio file from a file path or URL."""

    def __call__(self, file_path: str) -> dict[str, Union[torch.Tensor, int]]:
        """
        Args:
            file_path: Path or URL of the audio file.
        Returns:
            A dictionary containing the loaded audio tensor and the sample rate.
        """
        try:
            audio_tensor, sample_rate = torchaudio.load(file_path)
            return {"audio": audio_tensor, "sample_rate": sample_rate}
        except Exception as e:
            raise ValueError(f"Error loading audio: {e}")

    @staticmethod
    def get_spec():
        """Returns the node specification."""
        spec_file = os.path.join(os.path.dirname(__file__), 'audio_node.json')
        with open(spec_file, 'r', encoding='utf-8') as f:
            spec = json.load(f)
        return spec