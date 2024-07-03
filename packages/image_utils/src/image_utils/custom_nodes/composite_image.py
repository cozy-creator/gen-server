import traceback

import torch


def composite_images(background, foreground, mask=None, roi=None, resize=True):
    """
    Composites two images

    Parameters:
    background (torch.Tensor): The background image tensor with shape (C, H, W).
    foreground (torch.Tensor): The foreground image tensor with shape (C, H, W).
    mask (torch.Tensor, optional): An optional mask tensor with shape (H, W) or (1, H, W). Default is None.
    roi (tuple, optional): A tuple (x, y, width, height) specifying the region of interest for the composite. Default is None.
    resize (bool): Whether to resize the foreground and mask to match the background size. Default is True.

    Returns:
    torch.Tensor: The composited image tensor.
    """

    try:
        # Ensure all tensors are on the same device
        device = background.device
        foreground = foreground.to(device)
        if mask is not None:
            mask = mask.to(device)

        # Ensure tensors are in float range [0, 1]
        if background.max() > 1.0:
            background = background.float() / 255.0
        if foreground.max() > 1.0:
            foreground = foreground.float() / 255.0

        # Ensure images have alpha channels (4th channel)
        if background.shape[0] == 3:
            alpha_channel = torch.ones(
                1, background.shape[1], background.shape[2], device=device
            )
            background = torch.cat([background, alpha_channel], dim=0)
        if foreground.shape[0] == 3:
            alpha_channel = torch.ones(
                1, foreground.shape[1], foreground.shape[2], device=device
            )
            foreground = torch.cat([foreground, alpha_channel], dim=0)

        # Resize foreground and mask if necessary
        if resize and (
            background.shape[1] != foreground.shape[1]
            or background.shape[2] != foreground.shape[2]
        ):
            foreground = torch.nn.functional.interpolate(
                foreground.unsqueeze(0),
                size=background.shape[1:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            if mask is not None:
                mask = (
                    torch.nn.functional.interpolate(
                        mask.unsqueeze(0).unsqueeze(0),
                        size=background.shape[1:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )

        # Determine the alpha channel to use
        if mask is None:
            alpha = foreground[3:4, :, :]
        else:
            alpha = mask.unsqueeze(0) if mask.ndimension() == 2 else mask

        # Apply ROI if provided
        if roi:
            x, y, w, h = roi
            background_region = background[:, y : y + h, x : x + w]
            foreground_region = foreground[:, y : y + h, x : x + w]
            alpha_region = alpha[:, y : y + h, x : x + w]

            # Ensure the regions are the same size
            if (
                background_region.shape[1] != foreground_region.shape[1]
                or background_region.shape[2] != foreground_region.shape[2]
            ):
                raise ValueError("The ROI regions must have the same dimensions.")
        else:
            background_region = background
            foreground_region = foreground
            alpha_region = alpha

        # Perform compositing
        composite_region = (
            background_region[:3, :, :] * (1 - alpha_region)
            + foreground_region[:3, :, :] * alpha_region
        )

        # Place composite_region back into the background if ROI is used
        if roi:
            composite = background.clone()
            composite[:3, y : y + h, x : x + w] = composite_region
        else:
            composite = composite_region

        return composite

    except Exception as e:
        print(traceback.format_exc())
        print(f"Error processing images: {e}")
        return None
