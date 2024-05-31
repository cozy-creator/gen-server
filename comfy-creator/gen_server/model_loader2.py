import struct
import json
import time
import os

def parse_safetensors_header(file_path):
    start_time = time.time()  # Start the performance timer
    # Open the file in binary mode
    with open(file_path, 'rb') as file:
        length_of_header_bytes = file.read(8)
        # Interpret the bytes as a little-endian unsigned 64-bit integer
        length_of_header = struct.unpack('<Q', length_of_header_bytes)[0]
        header_bytes = file.read(length_of_header)
        metadata = json.loads(header_bytes)
    end_time = time.time()  # End the performance timer
    print(f"Time taken to parse metadata: {end_time - start_time} seconds")
    return metadata


def load_all_safetensors_headers(directory):
    headers = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.safetensors'):
                file_path = os.path.join(root, file)
                header = parse_safetensors_header(file_path)
                headers[file_path] = header
    return headers


model_type = {
    "controlnet": "controlnet",
    "conditioner": "sdxl",
    "cond_stage_model": "sd1.5",
    "lora": "lora",
}


def identify_model_type(metadata):
    for key in metadata.keys():
        for model in model_type.keys():
            if model in key:
                return model_type[model]
    return "Unknown"


with open('headers.md', 'w') as file:
    headers = load_all_safetensors_headers('./models')
    for file_path, metadata in headers.items():
        keys = metadata.keys()
        
        file.write(f"File: {file_path}\n")
        file.write(f"Model Type: {identify_model_type(metadata)}\n")
        file.write('\n'.join(keys))
        file.write('\n\n')


