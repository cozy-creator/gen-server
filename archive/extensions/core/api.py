from aiohttp import web
from core_library.uploader import S3Uploader
from gen_server.settings import S3Config
import blake3

async def upload_image(request):
    # Access request data (image file)
    data = await request.post()
    image_data = data['image'].file.read()

    # Initialize S3Uploader
    s3_config = S3Config()
    uploader = S3Uploader(s3_config)

    # Generate a unique filename
    filename = f"{blake3.blake3(image_data).hexdigest()}.png"
    key = f"uploads/{filename}"  # Assuming an "uploads" folder in your bucket

    # Upload the image
    try:
        uploader.upload(image_data, key, "image/png")
    except Exception as e:
        # Handle upload errors
        return web.json_response({"error": str(e)}, status=500)

    # Get the uploaded image URL
    image_url = uploader.get_uploaded_file_url(key)

    return web.json_response({"image_url": image_url})