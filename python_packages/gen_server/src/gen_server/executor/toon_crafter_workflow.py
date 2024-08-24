# ==== The poseable character workflow ====

# Step 1: load an SDXL model into memory

# Step 2: load two reference-images and prepare them for Clip vision

# Step 3: load ip adapter model; combine it (???) into SDXL model with partial strength

# Step 4: load an open-pose image, and a depth-mask image

# Step 5: load an open-pose and depth controlnet, then run the images though it
# and add it to SDXL, with partial strength

# Step 6: add text-conditioning to the SDXL model, and run a diffuserion on it on an empty
# latent image

# Result 1: an image, conditioned on pose, depth-map, character reference, and text-prompt

# Step 7: use auto-segmentation to select the face-area

# Step 8: run an image-to-image inpainting / denoising in order to clean this area up

# Result 2: a more cleaned-up version of the first image

# Step 9: remove the background (just foreground)

# Step 10: load an SDXL model, then condition it on text, and run it to generate a background

# Step 11: upscale the image to match the dimensions of the foreground, and add blur

# Step 12: composite together background + foreground

# Result 3: a final image, with background + foreground, both cleaned up


# Simplification for user

# user picks a pose
# user picks a depth map
# user picks a character (set of images)
# user picks

