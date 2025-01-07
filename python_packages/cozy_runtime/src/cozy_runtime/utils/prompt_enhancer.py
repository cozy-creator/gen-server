import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GenerationConfig, LogitsProcessor, LogitsProcessorList

class CustomLogitsProcessor(LogitsProcessor):
    def __init__(self, bias: float):
        super().__init__()
        self.bias = bias

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if len(input_ids.shape) == 2:
            last_token_id = input_ids[0, -1]
            self.bias[last_token_id] = -1e10

            return scores + self.bias
        
class PromptEnhancer:
    def __init__(self):
        self.styles = {
            "cinematic": "cinematic film still of {prompt}, highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain",
            "anime": "anime artwork of {prompt}, anime style, key visual, vibrant, studio anime, highly detailed",
            "photographic": "cinematic photo of {prompt}, 35mm photograph, film, professional, 4k, highly detailed",
            "comic": "comic of {prompt}, graphic illustration, comic art, graphic novel art, vibrant, highly detailed",
            "lineart": "line art drawing {prompt}, professional, sleek, modern, minimalist, graphic, line art, vector graphics",
            "pixelart": "pixel-art {prompt}, low-res, blocky, pixel art style, 8-bit graphics",
            "pony": "score_9, score_8_up, score_7_up, {prompt}",
            "masterpiece": "(masterpiece), (best quality), (ultra-detailed), {prompt}, illustration, disheveled hair, detailed eyes, perfect composition, moist skin, intricate details, earrings",
        }

        self.words = [
            "aesthetic", "astonishing", "beautiful", "breathtaking", "composition", 
            "contrasted", "epic", "moody", "enhanced", "exceptional", "fascinating",
            "flawless", "glamorous", "glorious", "illumination", "impressive", 
            "improved", "inspirational", "magnificent", "majestic", "hyperrealistic",
            "smooth", "sharp", "focus", "stunning", "detailed", "intricate", 
            "dramatic", "high", "quality", "perfect", "light", "ultra", "highly", 
            "radiant", "satisfying", "soothing", "sophisticated", "stylish", 
            "sublime", "terrific", "touching", "timeless", "wonderful", 
            "unbelievable", "elegant", "awesome", "amazing", "dynamic", "trendy"
        ]

        self.word_pairs = ["highly detailed", "high quality", "enhanced quality", "perfect composition", "dynamic light"] 

        self.tokenizer = GPT2Tokenizer.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion")
        self.model = GPT2LMHeadModel.from_pretrained("Gustavosta/MagicPrompt-Stable-Diffusion", torch_dtype=torch.float16).to("cuda")
        self.model.eval()

    def enhance_prompt(self, prompt: str, style: str = "cinematic") -> str:
        # Apply style to prompt
        prompt = self.styles[style].format(prompt=prompt)

        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        token_count = inputs["input_ids"].shape[1]
        max_new_tokens = 50 - token_count

        # Prepare logits processor
        word_ids = [self.tokenizer.encode(word, add_prefix_space=True)[0] for word in self.words]
        bias = torch.full((self.tokenizer.vocab_size,), -float("Inf")).to("cuda")
        bias[word_ids] = 0
        processor = CustomLogitsProcessor(bias)
        processor_list = LogitsProcessorList([processor])

        # Generate enhanced prompt
        generation_config = GenerationConfig(
            penalty_alpha=0.7,
            top_k=50,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.eos_token_id,
            pad_token=self.model.config.pad_token_id,
            do_sample=True,
        )

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                generation_config=generation_config,
                logits_processor=processor_list,
            )

        # Extract generated part and enhance prompt
        output_tokens = [self.tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in generated_ids]
        input_part, generated_part = output_tokens[0][: len(prompt)], output_tokens[0][len(prompt):]
        pairs, words = self.find_and_order_pairs(generated_part)
        formatted_generated_part = pairs + ", " + words
        return input_part + ", " + formatted_generated_part
    
    def find_and_order_pairs(self, s: str) -> tuple[str, str]:
        words = s.split()
        found_pairs = []
        for pair in self.word_pairs:
            pair_words = pair.split()
            if pair_words[0] in words and pair_words[1] in words:
                found_pairs.append(pair)
                words.remove(pair_words[0])
                words.remove(pair_words[1])
        ordered_pairs = ", ".join(found_pairs)
        remaining_s = ", ".join(words)
        return ordered_pairs, remaining_s
