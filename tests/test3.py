import msgpack
import requests
import websockets
import asyncio
import os
import json
import sseclient


# http://localhost:8881

class CozyClient:
    def __init__(self, base_url="http://localhost:8881", api_key=None):
        self.base_url = base_url
        self.headers = {
            'Content-Type': 'application/vnd.msgpack',
            'X-API-Key': api_key
        } if api_key else {'Content-Type': 'application/vnd.msgpack'}

    def generate(self, params):
        """Synchronous generation request"""
        url = f"{self.base_url}/v1/jobs/submit"
        packed_data = msgpack.packb(params)
        response = requests.post(url, data=packed_data, headers=self.headers)

        # Check for HTTP errors
        response.raise_for_status()

        # Parse the response as JSON
        try:
            unpacked_data = json.loads(response.content)
        except json.JSONDecodeError as e:
            print("Error decoding JSON response:", response.content)
            raise ValueError("Malformed server response") from e

        return unpacked_data

    async def generate_stream(self, params):
        """Asynchronous SSE streaming generation"""
        # Submit the job
        submit_url = f"{self.base_url}/v1/jobs/submit"
        packed_data = msgpack.packb(params)
        submit_response = requests.post(submit_url, data=packed_data, headers=self.headers)
        submit_response.raise_for_status()

        # Parse the job submission response (JSON)
        job_data = json.loads(submit_response.content)
        job_id = job_data.get('id')
        if not job_id:
            raise ValueError("Job ID missing in server response")

        # Open the SSE stream
        stream_url = f"{self.base_url}/v1/jobs/{job_id}/stream"
        stream_response = requests.get(stream_url, stream=True, headers=self.headers)
        print("Stream response Content:", stream_response.content)
        stream_response.raise_for_status()

        # Use SSEClient for streaming events
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
  "ebara-pony-xl"
]
# Example usage
async def main():
    client = CozyClient(api_key=os.getenv("COZY_API_KEY"))
    
    params = {
        "model": "playground",
        "positive_prompt": "A anime man fighting a peanut butter under the sea in space",
        "negative_prompt": "",
        "num_outputs": 1,
        # "random_seed": 43,
        "aspect_ratio": "1/1",
        "output_format": "png"
    }

    # Synchronous request
    result = client.generate(params)
    print("Sync result:", result)

    # Streaming request
    # async for event in client.generate_stream(params):
    #     print("Stream event:", event)

if __name__ == "__main__":
    asyncio.run(main())