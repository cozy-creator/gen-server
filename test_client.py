import asyncio
import json
import struct
from PIL import Image
import io
import os
from datetime import datetime

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
            try:
                # Read total message size
                total_size_data = await self.reader.read(4)
                if not total_size_data:
                    break
                total_size = struct.unpack("!I", total_size_data)[0]
                print(f"Total message size: {total_size}")

                # Read model ID size
                model_id_size_data = await self.reader.read(4)
                model_id_size = struct.unpack("!I", model_id_size_data)[0]
                print(f"Model ID size: {model_id_size}")

                # Read model ID
                model_id_bytes = await self.reader.read(model_id_size)
                model_id = model_id_bytes.decode('utf-8')
                print(f"Model ID: {model_id}")

                # Read the actual image data
                remaining_size = total_size - 4 - model_id_size  # Subtract model ID size header and model ID
                print(f"Expected image size: {remaining_size}")

                # Read image data in chunks
                image_data = bytearray()
                bytes_read = 0
                chunk_size = 8192  # 8KB chunks

                while bytes_read < remaining_size:
                    chunk = await self.reader.read(min(chunk_size, remaining_size - bytes_read))
                    if not chunk:
                        break
                    image_data.extend(chunk)
                    bytes_read += len(chunk)

                print(f"Actually read {bytes_read} bytes of image data")

                if bytes_read < remaining_size:
                    print(f"Warning: Incomplete read. Expected {remaining_size}, got {bytes_read}")

                try:
                    # Convert to bytes before creating BytesIO
                    image = Image.open(io.BytesIO(bytes(image_data)))
                    # Force load the entire image
                    image.load()
                    yield image
                except Exception as e:
                    print(f"Error loading image: {e}")
                    print(f"First 100 bytes of image data: {bytes(image_data)[:100].hex()}")

            except Exception as e:
                print(f"Error receiving data: {e}")
                import traceback
                traceback.print_exc()
                break

    async def close(self):
        self.writer.close()
        await self.writer.wait_closed()

async def main():
    # Create the generated_images directory if it doesn't exist
    output_dir = "generated_images"
    os.makedirs(output_dir, exist_ok=True)

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

    print("Sending request...")
    await client.send_request(request_data)

    print("Waiting for images...")
    image_count = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    async for image in client.receive_response():
        image_count += 1
        # Include timestamp and prompt in filename (shortened prompt to avoid too long filenames)
        prompt_slug = request_data["positive_prompt"][:30].replace(" ", "_").replace("/", "_")
        output_path = os.path.join(
            output_dir, 
            f"{timestamp}_{prompt_slug}_{image_count}.png"
        )
        
        try:
            image.save(output_path, "PNG")
            print(f"Saved image {image_count} to {output_path}")
        except Exception as e:
            print(f"Error saving image: {e}")
            import traceback
            traceback.print_exc()

    print("Done receiving images")
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())