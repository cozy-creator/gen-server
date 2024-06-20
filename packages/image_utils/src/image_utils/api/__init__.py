import os
import io
from typing import Optional, Dict, Union, Any, Iterable, List

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
        self.s3_client = comfy_config.s3.get("client", None)

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
                filename = f"{blake3.blake3(content).hexdigest()}{os.path.splitext(part.filename)[1]}"  # Assuming PNG files

                # Upload to S3 (if S3 client is provided)
                if self.s3_client:
                    try:
                        # Upload the file to the specified folder
                        key = f'{comfy_config.s3["folder"]}/{filename}' if comfy_config.s3["folder"] else f'{filename}'

                        # presigned_url = self.s3_client.generate_presigned_url(
                        # ClientMethod='put_object',
                        # Params={'Bucket': comfy_config.s3['bucket_name'], 'Key': key},
                        # ExpiresIn=3600,  # URL expiration time in seconds
                        # HttpMethod="PUT"
                        # )

                        self.s3_client.put_object(Bucket=comfy_config.s3["bucket_name"], Key=key, Body=content, ACL="public-read")
                        
                        # # Generate and append the image URL
                        image_url = f"{comfy_config.s3['url']}/{key}"
                        
                        # Return a success response with the URL
                        print("Successfully Uploaded Image")
                        return web.json_response({'success': True, 'url': image_url})
                    except Exception as e:
                        # Handle errors appropriately
                        print(f"Error uploading to S3: {e}")
                        return web.json_response({'success': False, 'error': str(e)})
                        

    async def set_public_acl(self, request: web.Request) -> web.Response:
        """
        Sets the ACL of the uploaded object to public-read.
        """
        data = await request.json()
        key = data.get('key')

        if not key:
            return web.json_response({'success': False, 'error': 'Key is missing'})

        try:
            self.s3_client.put_object_acl(Bucket=comfy_config.s3["bucket_name"], Key=key, ACL='public-read')
            return web.json_response({'success': True})
        except Exception as e:
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
        
    
    async def serve_local_file(self, request: web.Request) -> web.Response:
        """
        Serves a static file from the local filesystem.
        """
        file_path = request.match_info.get("file_path")  # Get the file path from the URL

        file_path = os.path.join(comfy_config.workspace_dir, "assets", file_path)

        if os.path.exists(file_path):
            # Read the file content in bytes
            with open(file_path, "rb") as f:
                file_content = f.read()

            # Get the content type dynamically
            content_type = get_content_type(file_path)

            # Create the response
            return web.Response(
                body=file_content, content_type=content_type, status=200
            )
        else:
            return web.Response(text="File not found", status=404)
        
        
def get_content_type(file_path):
    """
    Determines the content type based on the file extension.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    content_types = {
        '.html': 'text/html',
        '.css': 'text/css',
        '.js': 'application/javascript',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.ico': 'image/x-icon',
    }
    return content_types.get(file_extension, 'application/octet-stream')


# Load Files Locally instead


def file_handler() -> List[web.RouteDef]:
    """
    Creates a list of RouteDef instances containing route information for image uploads.
    """
    # Create a handler instance
    handler = FileHandler()

    routes: List[web.RouteDef] = [
        web.post('/upload', handler.handle_upload),
        web.post('/set-public-acl', handler.set_public_acl),
        web.get('/list', handler.list_files),
        web.get('/download/{file_key}', handler.download_file),
        web.get('/files/{file_path}', handler.serve_local_file)
    ]

    return routes