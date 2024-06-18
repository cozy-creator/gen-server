import os
import io
from typing import Optional, Dict, Union, Any

import aiohttp
from aiohttp import web
import blake3

from gen_server.globals import comfy_config
import boto3
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


# Create an AioHttp handler class
class FileHandler:
    """
    AioHttp handler for uploading images.
    """

    def __init__(self):
        self.s3_client = comfy_config.s3["client"]

    async def handle_upload(self, request: web.Request) -> web.Response:
        """
        Handles image upload requests.
        """
        reader = await request.multipart()

        # Initialize variables to store content and filename
        content = None
        filename = None

        # Iterate over parts
        while True:
            part = await reader.next()
            if not part:
                break
            if part.name == 'file':  # Assuming the file field is named "file"
                # Read the content
                content = await part.read()

                # Generate a unique filename based on hash
                filename = f"{blake3.blake3(content).hexdigest()}.png"  # Assuming PNG files

                # Upload to S3 (if S3 client is provided)
                if self.s3_client:
                    try:
                        # Upload the file to the specified folder
                        key = f'{comfy_config.s3["folder"]}/{filename}' if comfy_config.s3["folder"] else f'{filename}'

                        presigned_url = self.s3_client.generate_presigned_url(
                        ClientMethod='put_object',
                        Params={'Bucket': comfy_config.s3['bucket_name'], 'Key': key, 'ACL': 'public-read', 'ContentType': 'image/png'},
                        ExpiresIn=3600  # URL expiration time in seconds
                        )

                        # self.s3_client.put_object(Bucket=comfy_config.s3["bucket_name"], Key=key, Body=content, ACL="public-read")
                        
                        # # Generate and append the image URL
                        # image_url = f"{comfy_config.s3['url']}/{key}"
                        
                        # Return a success response with the URL
                        print("Successfully uploaded file")
                        return web.json_response({'success': True, 'url': presigned_url})
                    except Exception as e:
                        # Handle errors appropriately
                        print(f"Error uploading to S3: {e}")
                        return web.json_response({'success': False, 'error': str(e)})
                        

                    
    async def list_files(self, request: web.Request) -> web.Response:
        if comfy_config.s3["folder"]:
            folder = comfy_config.s3["folder"]
        else:
            folder = request.match_info['folder']  # Get the folder path from URL
        
        try:
            # List objects in the specified folder (prefix)
            response = self.s3_client.list_objects_v2(Bucket=comfy_config.s3["bucket_name"], Prefix=folder)
            
            # Extract URLs of the files
            file_urls = []
            for obj in response.get('Contents', []):
                # Generate the URL for each object in the bucket
                object_url = f"https://{comfy_config.s3['bucket_name']}.{urlparse(self.s3_client.meta.endpoint_url).hostname}/{obj['Key']}"
                file_urls.append(object_url)
            
            return web.json_response({'success': True, 'files': file_urls})
        
        except Exception as e:
            return web.json_response({'success': False, 'error': str(e)})
        

    async def download_file(self, request: web.Request) -> web.FileResponse:
        file_key = request.match_info['file_key']  # Get the file key from URL

        key = f'{comfy_config.s3["folder"]}/{file_key}' if comfy_config.s3["folder"] else f'{file_key}'
        
        try:
            # Generate a presigned URL for downloading the file
            url = self.s3_client.generate_presigned_url(
                ClientMethod='get_object',
                Params={'Bucket': comfy_config.s3['bucket_name'], 'Key': key},
                ExpiresIn=3600  # URL expiration time in seconds
            )
            
            return web.json_response({'success': True, 'download_url': url})
        
        except Exception as e:
            return web.json_response({'success': False, 'error': str(e)})


# Load Files Locally instead


def file_handler() -> web.Application:
    """
    Creates a list of tuples containing route information for image uploads.
    """

    # Create a handler instance
    handler = FileHandler()

    routes = [
        ('POST', '/upload', handler.handle_upload),
        ('GET', '/list', handler.list_files),
        ('GET', '/download/{file_key}', handler.download_file)
    ]

    return routes