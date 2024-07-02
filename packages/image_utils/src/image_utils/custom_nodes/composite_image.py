import torch
import torchvision.transforms.functional as TF


def composite_images(
    background: torch.Tensor,
    foreground: torch.Tensor,
    mask: torch.Tensor = None,
    region=None,
    resize=True,
):
    """
    Composites two images using PyTorch tensors, handling different sizes, modes, and transparency, with an optional mask and region.

    Parameters:
    background (torch.Tensor): The background image tensor with shape (C, H, W).
    foreground (torch.Tensor): The foreground image tensor with shape (C, H, W).
    mask (torch.Tensor, optional): An optional mask tensor with shape (H, W) or (1, H, W). Default is None.
    region (tuple, optional): A tuple (x, y, width, height) specifying the region of interest for the composite. Default is None.
    resize (bool): Whether to resize the foreground and mask to match the background size. Default is True.

    Returns:
    torch.Tensor: The composited image tensor.
    """

    try:
        # Ensure tensors are in float range [0, 1]
        background = (
            background.float() / 255.0 if background.max() > 1.0 else background.float()
        )
        foreground = (
            foreground.float() / 255.0 if foreground.max() > 1.0 else foreground.float()
        )

        # Ensure images have alpha channels (4th channel)
        if background.shape[0] == 3:
            background = torch.cat(
                [background, torch.ones(1, *background.shape[1:])], dim=0
            )
        if foreground.shape[0] == 3:
            foreground = torch.cat(
                [foreground, torch.ones(1, *foreground.shape[1:])], dim=0
            )

        # Resize foreground and mask if necessary
        if resize and background.shape[1:] != foreground.shape[1:]:
            foreground = TF.resize(foreground, *background.shape[1:])
            if mask is not None:
                mask = TF.resize(mask, *background.shape[1:])

        # Determine the alpha channel to use
        if mask is None:
            alpha = foreground[3:4, :, :]
        else:
            alpha = mask.unsqueeze(0) if mask.ndimension() == 2 else mask

        # Apply region if provided
        if region:
            x, y, w, h = region
            background_region = background[:, y : y + h, x : x + w]
            foreground_region = foreground[:, y : y + h, x : x + w]
            alpha_region = alpha[:, y : y + h, x : x + w]
        else:
            background_region = background
            foreground_region = foreground
            alpha_region = alpha

        # Perform compositing
        composite_region = (
            background_region[:3, :, :] * (1 - alpha_region)
            + foreground_region[:3, :, :] * alpha_region
        )

        # Place composite_region back into the background if region is used
        if region:
            composite_tensor = background.clone()
            composite_tensor[:3, y : y + h, x : x + w] = composite_region
        else:
            composite_tensor = composite_region

        return composite_tensor

    except Exception as e:
        print(f"Error processing images: {e}")
        return None
