from boto3 import session

from gen_server.settings import settings


def get_client():
    return session.Session().client(
        's3',
        region_name=settings.s3.region_name,
        endpoint_url=settings.s3.endpoint_url,
        aws_access_key_id=settings.s3.access_key,
        aws_secret_access_key=settings.s3.secret_access_key
    )


def create_multipart_upload(*, client, key: str, mime_type: str):
    upload = client.create_multipart_upload(Bucket=settings.s3.bucket_name, Key=key, ContentType=mime_type)
    return upload


def upload_chunk(*, client, upload_id: str, key: str, chunk: bytes, part_number: int):
    return client.upload_part(
        Bucket=settings.s3.bucket_name,
        PartNumber=part_number,
        UploadId=upload_id,
        Key=key,
        Body=chunk
    )


def complete_multipart_upload(*, client, parts, upload_id: str, key: str):
    return client.complete_multipart_upload(
        Key=key,
        UploadId=upload_id,
        Bucket=settings.s3.bucket_name,
        MultipartUpload={'Parts': parts}
    )


def upload_raw_bytes(*, client, key: str, data: bytes, mime_type: str):
    return client.put_object(Bucket=settings.s3.bucket_name, Key=key, Body=data, ContentType=mime_type)


def get_uploaded_file_url(key: str):
    return f"{settings.s3.endpoint_url}/{settings.s3.bucket_name}/{key}"
