import asyncio
import logging
import multiprocessing
from queue import Empty
import queue
from PIL import Image, PngImagePlugin
import torch
from multiprocessing.connection import Connection
from ..base_types.pydantic_models import RunCommandConfig

from ..utils.file_handler import FileHandler, get_file_handler
from ..utils.image import tensor_to_pil

logger = logging.getLogger(__name__)



async def start_io_worker(
    tensor_queue: queue.Queue,
    cozy_config: RunCommandConfig
):
    file_handler = get_file_handler(cozy_config)
    
    print("IO worker started", flush=True)
    
    while True:
        try:
            try:
                # Unfortunately the inter-process communication queue is not async
                # this is a blocking call
                tensor_batch, response_conn = tensor_queue.get(timeout=1)  # Wait for up to 1 second
                
                # Use asyncio.to_thread to run the blocking queue.get in a separate thread
                # tensor_batch, response_conn = await asyncio.to_thread(tensor_queue.get, timeout=1)
                
                if tensor_batch is None:
                    print("Received None tensor_batch, signaling end of processing")
                    # GPU worker is done with this connection
                    response_conn.send(None)
                    continue
                
                print(f"Received new batch. Tensor shape: {tensor_batch.shape}")

                await upload_batch(file_handler, tensor_batch, response_conn)
                tensor_queue.task_done()

                await asyncio.sleep(0) # Yield control to other async tasks
            
            except Empty:
                await asyncio.sleep(0) # Yield control to other async tasks
                pass # No new job, continue with ongoing tasks
        
        except Exception as e:
            logger.error(f"Unexpected error in io-worker: {str(e)}")
    
    # Do we need to wait for all remaining tasks to complete before shutting down?
    logger.info("IO-worker shut down complete")


async def upload_batch(
    file_handler: FileHandler,
    tensor_batch: torch.Tensor,
    response_conn: Connection
):
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Description", "Generated by gen_server")
    metadata.add_text("Author", "gen_server")
    
    pil_images = tensor_to_pil(tensor_batch)
    
    print("Starting to upload PNG files")
    async for file_url in file_handler.upload_png_files(pil_images, metadata):
        print(f"File uploaded successfully. URL: {file_url}")
        response_conn.send(file_url)
    
