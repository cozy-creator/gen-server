import msgpack
import requests
import websockets
import asyncio
import os

class CozyClient:
    def __init__(self, base_url="http://localhost:8881", api_key=None):
        self.base_url = base_url
        self.headers = {
            'Content-Type': 'application/msgpack',
            'X-API-Key': api_key
        } if api_key else {'Content-Type': 'application/msgpack'}

    def generate(self, params):
        """Synchronous generation request"""
        url = f"{self.base_url}/v1/jobs/submit"
        packed_data = msgpack.packb(params)
        response = requests.post(url, data=packed_data, headers=self.headers)
        return msgpack.unpackb(response.content)

    async def generate_stream(self, params):
        """Asynchronous streaming generation"""
        url = f"{self.base_url}/v1/jobs/stream"
        packed_data = msgpack.packb(params)
        response = requests.post(f"{self.base_url}/v1/jobs/submit", 
                               data=packed_data, 
                               headers=self.headers)
        job_data = msgpack.unpackb(response.content)
        job_id = job_data['id']
        
        ws_url = f"ws://localhost:8881/v1/jobs/{job_id}/stream"
        async with websockets.connect(ws_url) as websocket:
            while True:
                msg = await websocket.recv()
                if msg == b'END':
                    break
                yield msgpack.unpackb(msg)

# Example usage
async def main():
    client = CozyClient(api_key=os.getenv("COZY_API_KEY"))
    
    params = {
        "model": "sd1.5-runwayml",
        "positive_prompt": "a cat sitting on a windowsill",
        "negative_prompt": "blurry, low quality",
        "num_outputs": 1,
        "random_seed": 42,
        "aspect_ratio": "1/1",
        "output_format": "png"
    }

    # Synchronous request
    result = client.generate(params)
    print("Sync result:", result)

    # Streaming request
    async for event in client.generate_stream(params):
        print("Stream event:", event)

if __name__ == "__main__":
    asyncio.run(main())