import msgpack
import requests
import websockets
import asyncio
import os
import json
import sseclient
# from diffusers.utils import load_image
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL

# http://localhost:8881

class CozyClient:
    def __init__(self, base_url="http://localhost:8881", api_key=None):
        self.base_url = base_url
        self.headers = {
            'Content-Type': 'application/vnd.msgpack',
            'X-API-Key': api_key
        } if api_key else {'Content-Type': 'application/vnd.msgpack'}

    def generate(self, params, loras=None):
        """
        Synchronous generation request with optional LoRA support
        
        Args:
            params: Base generation parameters
            loras: Optional list of dicts containing {'url': 'https://...', 'scale': 0.75}
        """
        if loras:
            params["loras"] = loras

        url = f"{self.base_url}/v1/jobs/submit"
        packed_data = msgpack.packb(params)
        response = requests.post(url, data=packed_data, headers=self.headers)

        response.raise_for_status()

        try:
            unpacked_data = json.loads(response.content)
        except json.JSONDecodeError as e:
            print("Error decoding JSON response:", response.content)
            raise ValueError("Malformed server response") from e

        return unpacked_data

    async def generate_stream(self, params, lora=None):
        """
        Asynchronous SSE streaming generation with optional LoRA support
        """
        if lora:
            params["lora"] = lora

        submit_url = f"{self.base_url}/v1/jobs/submit"
        packed_data = msgpack.packb(params)
        submit_response = requests.post(submit_url, data=packed_data, headers=self.headers)
        submit_response.raise_for_status()

        # Rest of your existing streaming code...
        job_data = json.loads(submit_response.content)
        job_id = job_data.get('id')
        if not job_id:
            raise ValueError("Job ID missing in server response")

        stream_url = f"{self.base_url}/v1/jobs/{job_id}/stream"
        stream_response = requests.get(stream_url, stream=True, headers=self.headers)
        stream_response.raise_for_status()

        client = sseclient.SSEClient(stream_response)
        for event in client.events():
            yield json.loads(event.data)


[
  "flux.1-schnell-fp8",
  "sd3.5-large-int8",
  "playground2.5",
  "pony.v6",
  "cyberrealistic.pony",
  "wai.ani.ponyxl",
  "illustrious.xl",
  "real.dream.pony",
  "pony.realism",
  "babes_by_stable_yogi.v4.xl",
  "ebara-pony-xl",
  "auraflow",
  "pony.realism"
]
# Example usage
async def main():
    client = CozyClient(api_key=os.getenv("COZY_API_KEY"))
    # A high-resolution, hyper-realistic photograph of an American corgi with a tri-color coat, wearing a blue bandana around its neck, sitting on a lush green lawn in a sunlit park, with a background of blooming flowers and tall trees, captured with a shallow depth of field, vibrant colors, and soft natural lighting.
    # score_9, score_8_up, score_7_up, BREAK, female, solo, clothing, hair, long hair, smile, fruit, food, sitting, anthro, looking at viewer, mammal, brown eyes, black hair, hi res, simple background, digital media (artwork), haplorhine

    params = {
        "model": "juggernaut-xl-v9",
        "positive_prompt": "anime, pastel color, 1girl, holding instrument electric guitar, playing instrument, pleated skirt, spotlight, standing, looking away, blush, gloom, feet out of frame, gibson les paul, guitar, hair ornament, cube hair ornament, blue eyes, pink long hair, pink track jacket, bangs, hair between eyes, one side up,",
        # "negative_prompt": "score_6_up, score_5_up, score_4_up, censored, deformed, bad hand, bad anatomy, cartoon",
        "num_outputs": 1,
        "random_seed": 1020,
        "aspect_ratio": "1/1",
        "output_format": "png",
        "use_lora_recommender": True
        # "enhance_prompt": True,
        # "style": "pony"
    }

    # https://civitai.com/api/download/models/12345
    # https://huggingface.co/ralux3/sdxl-lora/resolve/main/pytorch_lora_weights.safetensors?download=true

    lora_config = [
        {
            "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors?download=true",
            "scale": 1.0
        },
        # {
        #     "url": "https://huggingface.co/ostris/crayon_style_lora_sdxl/resolve/main/crayons_v1_sdxl.safetensors?download=true",
        #     "scale": 0.35
        # },
        # {
        #     "url": "https://civitai.com/api/download/models/129888?type=Model&format=SafeTensor",
        #     "scale": 0.75
        # },
        # {
        #     "url": "https://huggingface.co/ProomptEngineer/pe-balloon-diffusion-style/resolve/main/PE_BalloonStyle.safetensors?download=true",
        #     "scale": 0.75
        # },
        # {
        #     "url": "https://huggingface.co/Pclanglais/TintinIA/resolve/main/pytorch_lora_weights.safetensors?download=true",
        #     "scale": 1.0
        # },
        # {
        #     "url": "https://civitai.com/api/download/models/273591?type=Model&format=SafeTensor",
        #     "scale": 1.2
        # },
        # {
        #     "url": "https://civitai.com/api/download/models/317584?type=Model&format=SafeTensor",
        #     "scale": 0.73
        # }
    ]

    # Synchronous request
    result = client.generate(params)
    print("Sync result:", result)

    # Streaming request
    # async for event in client.generate_stream(params):
    #     print("Stream event:", event)

if __name__ == "__main__":
    asyncio.run(main())