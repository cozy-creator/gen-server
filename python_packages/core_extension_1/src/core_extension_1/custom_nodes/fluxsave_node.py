from gen_server.base_types import CustomNode
from typing import AsyncGenerator, Dict, Any

class SaveLoraNode(CustomNode):
    async def __call__(self, train_generator: AsyncGenerator[Dict[str, Any], None]):
        sample_images = []
        lora_files = []

        async for update in train_generator:
            if update['type'] == 'sample':
                sample_images.extend(update['paths'])
                yield {'type': 'sample_images', 'step': update['current'], 'paths': update['paths']}
            elif update['type'] == 'save':
                lora_files.append(update['path'])
                yield {'type': 'lora_file', 'step': update['current'], 'path': update['path']}
            elif update['type'] == 'step':
                yield update
            elif update['type'] == 'finished':
                break
            elif update['type'] == 'cancelled':
                print("In Here")
                yield update
                break

        if not update['type'] == 'cancelled':
            yield {
                'type': 'final_result',
                'sample_images': sample_images,
                'lora_files': lora_files,
                'final_lora': lora_files[-1] if lora_files else None
            }
        else:
            yield update