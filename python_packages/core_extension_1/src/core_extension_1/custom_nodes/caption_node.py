from gen_server.base_types import CustomNode
from typing import Dict, List, Any
import os
import shutil
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from gen_server.utils.paths import get_assets_dir, get_home_dir


class CustomCaptionNode(CustomNode):
    def __init__(self):
        super().__init__()
        self.captions = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     'microsoft/Florence-2-large', 
        #     trust_remote_code=True, 
        #     torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        # ).eval().to(self.device)
        # self.processor = AutoProcessor.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True)
        self.home_dir = get_home_dir()

    async def __call__(self, 
                       image_paths: List[str], 
                       captions: Dict[str, str] = None,
                       use_auto_captioning: bool = False,
                       output_directory: str = None) -> Dict[str, Any]:
        """
        Manage custom captions for images, with optional auto-captioning and processing.

        Args:
            image_paths (List[str]): List of paths to images.
            captions (Dict[str, str], optional): Dictionary of image paths and their captions.
            use_auto_captioning (bool): Whether to use auto-captioning for images without custom captions.
            output_directory (str, optional): Directory to save processed images and captions.

        Returns:
            Dict[str, Any]: Dictionary containing image captions and processed directory information.
        """
        result = {}
        
        # Sometimes, user will pass in a single directory that contains images.
        # If so, we need to get all the images in the directory and process them.
        if len(image_paths) == 1 and os.path.isdir(f"{self.home_dir}/{image_paths[0]}"):
            image_paths = [os.path.join(image_paths[0], f) for f in os.listdir(f"{self.home_dir}/{image_paths[0]}") if 
                           os.path.isfile(os.path.join(f"{self.home_dir}/{image_paths[0]}", f))]

        for image_path in image_paths:
            image_path = os.path.join(self.home_dir, image_path)
            image_name = os.path.basename(image_path)
            if captions and image_path in captions:
                self.captions[image_path] = captions[image_path]
            elif image_path not in self.captions:
                if use_auto_captioning:
                    # self.captions[image_path] = self.auto_caption_image(image_path)
                    continue
                else:
                    self.captions[image_path] = ""

            result[image_path] = {
                "name": image_name,
                "caption": self.captions[image_path]
            }
            print(f"Image: {image_name}, Caption: {self.captions[image_path]}")

        processed_directory = None
        if output_directory:
            processed_directory = self.process_images(image_paths, output_directory)

        return {
            "image_captions": result,
            "processed_directory": processed_directory
        }

    # def auto_caption_image(self, image_path: str) -> str:
    #     image = Image.open(image_path).convert("RGB")
    #     inputs = self.processor(text="<MORE_DETAILED_CAPTION>", images=image, return_tensors="pt").to(self.device)
        
    #     with torch.no_grad():
    #         generated_ids = self.model.generate(
    #             input_ids=inputs["input_ids"],
    #             pixel_values=inputs["pixel_values"],
    #             max_new_tokens=1024,
    #             num_beams=3,
    #             do_sample=False
    #         )
        
    #     generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    #     parsed_answer = self.processor.post_process_generation(
    #         generated_text, 
    #         task="<MORE_DETAILED_CAPTION>", 
    #         image_size=(image.width, image.height)
    #     )
        
    #     return parsed_answer["<MORE_DETAILED_CAPTION>"]

    def process_images(self, image_paths: List[str], output_directory: str) -> str:
        processed_directory = f"{output_directory}_processed"
        processed_directory = os.path.join(self.home_dir, processed_directory)
        os.makedirs(processed_directory, exist_ok=True)

        for image_path in image_paths:
            image_path = os.path.join(self.home_dir, image_path)
            image_file = os.path.basename(image_path)
            new_image_path = os.path.join(processed_directory, image_file)
            new_caption_path = os.path.join(processed_directory, f"{os.path.splitext(image_file)[0]}.txt")

            # Copy image to processed directory
            shutil.copy2(image_path, new_image_path)

            # Write caption to text file
            with open(new_caption_path, "w") as f:
                f.write(self.captions[image_path])

        print(f"Images processed and saved to {processed_directory}")

        return processed_directory
