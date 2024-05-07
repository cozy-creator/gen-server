import torch
import numpy as np
import PIL.Image

class tensor2vid():
    def __init__(self):
        pass
    
    def __call__(video: torch.Tensor, processor: VaeImageProcessor):
        batch_size, channels, num_frames, height, width = video.shape
        for batch_idx in range(batch_size):
            batch_vid = video[batch_idx].permute(1, 0, 2, 3)  # Reorder dimensions for processing
            batch_output = processor.postprocess(batch_vid, output_type="pil")
            yield batch_output  # Yield the PIL images directly
    
    def pil_to_np

# this is an example of how the executor can do the conversion automatically
def execute_video_processing(video_tensor, processor):
    pil_images = []
    np_images = []
    pt_images = []

    for pil_output in tensor2vid(video_tensor, processor):
        pil_images.extend(pil_output)  # Collect PIL images
        np_images.extend([np.array(img) for img in pil_output])  # Convert to NumPy arrays and collect
        pt_images.extend([torch.from_numpy(np.array(img)) for img in pil_output])  # Convert to PyTorch tensors and collect

    return {
        "pil": pil_images,
        "np": np.stack(np_images),
        "pt": torch.stack(pt_images)
    }