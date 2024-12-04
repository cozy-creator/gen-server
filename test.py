# from OmniGen import OmniGenPipeline
# from diffusers import DiffusionPipeline, StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline, AutoEncoderXL

# pipe = OmniGenPipeline.from_pretrained("Shitao/OmniGen-v1")

# # pipe.enable_model_cpu_offload()

# # Text to Image
# images = pipe(
#     prompt="A curly-haired man in a red shirt is drinking tea.", 
#     height=1024, 
#     width=1024, 
#     guidance_scale=2.5,
#     seed=0,
#     offload_model=True
# )
# images[0].save("example_t2i.png") 

# # # Multi-modal to Image
# # # In prompt, we use the placeholder to represent the image. The image placeholder should be in the format of <img><|image_*|></img>
# # # You can add multiple images in the input_images. Please ensure that each image has its placeholder. For example, for the list input_images [img1_path, img2_path], the prompt needs to have two placeholders: <img><|image_1|></img>, <img><|image_2|></img>.
# # images = pipe(
# #     prompt="A man in a black shirt is reading a book. The man is the right man in <img><|image_1|></img>.",
# #     input_images=["./imgs/test_cases/two_man.jpg"],
# #     height=1024, 
# #     width=1024,
# #     separate_cfg_infer=False,  # if OOM, you can set separate_cfg_infer=True 
# #     guidance_scale=3, 
# #     img_guidance_scale=1.6
# # )
# # images[0].save("example_ti2i.png")