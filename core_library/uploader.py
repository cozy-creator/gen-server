import os
from typing import Optional, Union

from boto3 import session

from gen_server.settings import S3Config


class S3Uploader:
    client: Optional[session.Session.client] = None
    upload_chunk_size = 5 * 1024 * 1024
    config: Optional[S3Config] = None

    def __init__(self, config: Optional[S3Config], upload_chunk_size: int = 5 * 1024 * 1024):
        self.upload_chunk_size = upload_chunk_size
        if config is not None:
            self.config = config
            self.client = session.Session().client(
                's3',
                region_name=config.region_name,
                endpoint_url=config.endpoint_url,
                aws_access_key_id=config.access_key,
                aws_secret_access_key=config.secret_access_key
            )

    def upload(self, file: Union[bytes, str], key: str, mime_type: str):
        if self.client is None:
            raise Exception("Client is not initialized")

        if isinstance(file, str):
            size = os.path.getsize(file)
        else:
            size = len(file)

        if size <= self.upload_chunk_size:
            if isinstance(file, str):
                with open(file, 'rb') as f:
                    return self.upload_raw_bytes(key, f.read(), mime_type)
            else:
                return self.upload_raw_bytes(key, file, mime_type)
        else:
            if isinstance(file, bytes):
                raise Exception("File size exceeds the upload chunk size")
            return self.upload_file_multipart(file, key, mime_type)

    def upload_file_multipart(self, file_path: str, key: str, mime_type: str):
        upload = self.create_multipart_upload(key=key, mime_type=mime_type)

        parts, part_number = [], 1
        with open(file_path, 'rb') as file:
            while True:
                chunk = file.read(self.upload_chunk_size)

                if not chunk:
                    break

                resp = self.upload_chunk(
                    key=key,
                    chunk=chunk,
                    part_number=part_number,
                    upload_id=upload['UploadId']
                )

                if resp is not None:
                    parts.append({'PartNumber': part_number, 'ETag': resp['ETag']})

                part_number += 1

            response = self.complete_multipart_upload(upload_id=upload['UploadId'], key=key, parts=parts)
        return response

    def create_multipart_upload(self, key: str, mime_type: str):
        if self.client is None:
            raise Exception("Client is not initialized")

        return self.client.create_multipart_upload(Bucket=self.config.bucket_name, Key=key, ContentType=mime_type)

    def upload_chunk(self, upload_id: str, key: str, chunk: bytes, part_number: int):
        if self.client is None:
            raise Exception("Client is not initialized")

        return self.client.upload_part(
            Bucket=self.config.bucket_name,
            PartNumber=part_number,
            UploadId=upload_id,
            Key=key,
            Body=chunk
        )

    def complete_multipart_upload(self, parts, upload_id: str, key: str):
        if self.client is None:
            raise Exception("Client is not initialized")

        return self.client.complete_multipart_upload(
            Key=key,
            UploadId=upload_id,
            Bucket=self.config.bucket_name,
            MultipartUpload={'Parts': parts}
        )

    def upload_raw_bytes(self, key: str, data: bytes, mime_type: str):
        if self.client is None:
            raise Exception("Client is not initialized")

        return self.client.put_object(Bucket=self.config.bucket_name, Key=key, Body=data, ContentType=mime_type)

    def get_uploaded_file_url(self, key: str):
        if self.client is None:
            raise Exception("Client is not initialized")

        return f"{self.config.endpoint_url}/{self.config.bucket_name}/{key}"
