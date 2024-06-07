from openai import OpenAI

openai = OpenAI(api_key = 'key')

class OpenAITextGeneration:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "model": (["gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0613", "gpt-4-turbo"],), 
                },
            "optional": {
                "max_tokens": ("INT", {"default": 0}),
                "temperature": ("INT", {"default": 1.0, "min": 0, "max": 2, "step": 0.01}),
                }
            }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"

    CATEGORY = "OpenAI"

    def generate_text(self, text, model, max_tokens=None, temperature=1.0):
        # Send the request to the OpenAI API
        try:
            response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
                    {"role": "user", "content": text}
            ],
            max_tokens=max_tokens,
            temperature=temperature
            )

            # Extract the generated text from the response
            generated_text = response.choices[0].message.content
            print(generated_text, "HERE")
        except Exception as e:
            raise ValueError(f"Error generating text: {e}")

        return (generated_text, )