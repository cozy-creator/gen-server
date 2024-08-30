from gen_server.base_types import CustomNode
from PIL import Image
import os
import shutil
from typing import Dict, Any
import torch
from transformers import AutoProcessor, AutoModelForCausalLM



class FLUXDataPrepNode(CustomNode):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            'microsoft/Florence-2-large', 
            trust_remote_code=True, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True)

    async def __call__(self, 
                       image_directory: str, 
                       custom_captions: Dict[str, Dict[str, str]],
                       use_auto_captioning: bool = False) -> dict[str, str]:
        processed_directory = f"{image_directory}_processed"
        os.makedirs(processed_directory, exist_ok=True)

        for image_file, image_info in custom_captions.items():
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_directory, image_file)
                new_image_path = os.path.join(processed_directory, image_file)
                new_caption_path = os.path.join(processed_directory, f"{os.path.splitext(image_file)[0]}.txt")

                # Copy image to processed directory
                shutil.copy2(image_path, new_image_path)

                # Write caption to text file
                with open(new_caption_path, "w") as f:
                    if use_auto_captioning and not image_info['caption']:
                        caption = self.auto_caption_image(image_path)
                    else:
                        caption = image_info['caption']
                    f.write(caption)

        return {"processed_directory": processed_directory}

    def auto_caption_image(self, image_path: str) -> str:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text="<MORE_DETAILED_CAPTION>", images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task="<MORE_DETAILED_CAPTION>", 
            image_size=(image.width, image.height)
        )
        
        return parsed_answer["<MORE_DETAILED_CAPTION>"]
