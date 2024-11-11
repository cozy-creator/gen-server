import asyncio
import json
import struct
from PIL import Image
import io

class TCPClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port

    async def connect(self):
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)

    async def send_request(self, data):
        json_data = json.dumps(data).encode()
        self.writer.write(json_data)
        await self.writer.drain()

    async def receive_response(self):
        while True:
            size_data = await self.reader.read(4)
            if not size_data:
                break

            size = struct.unpack("!I", size_data)[0]
            image_data = await self.reader.read(size)
            
            # Convert the bytes to a PIL Image
            image = Image.open(io.BytesIO(image_data))
            yield image

    async def close(self):
        self.writer.close()
        await self.writer.wait_closed()

async def main():
    client = TCPClient('127.0.0.1', 8881)
    await client.connect()

    # Sample request data
    request_data = {
        "model": "juggernaut-xl-v9",
        "num_outputs": 1,
        "positive_prompt": "A beautiful landscape with mountains and a lake",
        "negative_prompt": "No people, no buildings",
        "random_seed": 42,
        "aspect_ratio": "16/9"
    }

    await client.send_request(request_data)

    print("Waiting for images")
    image_count = 0
    async for image in client.receive_response():
        print("Done")
        print(image)
        image_count += 1
        image.save(f"received_image_{image_count}.png")
        print(f"Received and saved image {image_count}")

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())