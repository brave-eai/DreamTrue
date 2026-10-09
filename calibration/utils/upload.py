import abc
import base64
import hashlib
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed


class Uploader(abc.ABC):

    @abc.abstractmethod
    def upload(self, filepath: str, path_in_repo: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def exists(self, path_in_repo: str) -> bool:
        raise NotImplementedError

    @classmethod
    def from_url(cls, upload_id: str) -> 'Uploader':
        if upload_id == '':
            return _NullUploader()
        upload_dict = re.compile(r'^'
                                 r'(?:(?P<proto>[a-zA-Z][a-zA-Z0-9+.-]*)://)?'  # proto
                                 r'(?:(?P<user>[^:@/]+)(?::(?P<password>[^@/]*))?@)?'  # user:pass@
                                 r'(?P<url>[^/]*?)'  # host
                                 r'/(?P<path>.+)'  # path
                                 r'$').match(upload_id)
        assert upload_dict is not None
        upload_dict = {k: v for k, v in upload_dict.groupdict().items() if v is not None and v != ''}
        if upload_dict['proto'] == 'modelscope':
            return _ModelScopeUploader(upload_dict)
        elif upload_dict['proto'] == 'oss':
            return _OSSUploader(upload_dict)
        elif upload_dict['proto'] == 'file':
            return _FileUploader(upload_dict)
        else:
            raise NotImplementedError(f'Unsupported upload proto {upload_dict["proto"]}')


class _NullUploader(Uploader):

    def upload(self, filepath: str, path_in_repo: str) -> None:
        pass

    def exists(self, path_in_repo: str) -> bool:
        return False


class _ModelScopeUploader(Uploader):

    def __init__(self, upload_dict: dict):
        self.__endpoint = upload_dict.get('url', None)
        if self.__endpoint is not None and not self.__endpoint.startswith('http'):
            from modelscope.hub.constants import MODELSCOPE_URL_SCHEME
            self.__endpoint = MODELSCOPE_URL_SCHEME + self.__endpoint
        self.__token = upload_dict.get('user', None)
        path_parts = upload_dict['path'].split('/', maxsplit=2)
        self.__repo_id = '/'.join(path_parts[:2])
        self.__path_prefix = path_parts[2].strip('/') if len(path_parts) > 2 else ''

    def __full_path(self, path_in_repo: str) -> str:
        return f'{self.__path_prefix}/{path_in_repo}' if self.__path_prefix else path_in_repo

    def upload(self, filepath: str, path_in_repo: str) -> None:
        import random
        import time

        from modelscope import HubApi

        path_in_repo = self.__full_path(path_in_repo)

        class MyHubApi(HubApi):

            # Add retry logic to handle transient failures (e.g. commit conflicts when multiple processes upload to the same repo simultaneously).
            def __retry(self, fn, *args, **kwargs):
                attempt = 0
                while True:
                    try:
                        attempt += 1
                        return fn(*args, **kwargs)
                    except Exception as e:
                        if attempt < 11:
                            time.sleep(attempt - 1 + random.random())
                        else:
                            raise e

            def create_commit(self, *args, **kwargs):
                return self.__retry(super().create_commit, *args, **kwargs)

            def create_repo(self, *args, **kwargs):
                return self.__retry(super().create_repo, *args, **kwargs)

            def upload_file(self, *args, **kwargs):
                return self.__retry(super().upload_file, *args, **kwargs)

        MyHubApi(max_retries=32, endpoint=self.__endpoint, token=self.__token).upload_file(
            repo_id=self.__repo_id,
            path_or_fileobj=filepath,
            path_in_repo=path_in_repo,
            commit_message=f'upload {path_in_repo}',
            repo_type='dataset',
            disable_tqdm=False,
            buffer_size_mb=8,
        )

    def exists(self, path_in_repo: str) -> bool:
        from modelscope.hub.api import HubApi
        api = HubApi(max_retries=1, endpoint=self.__endpoint, token=self.__token)
        path_in_repo = self.__full_path(path_in_repo)
        page_number, page_size = 1, 100
        while True:
            files = api.get_dataset_files(
                repo_id=self.__repo_id,
                root_path=os.path.dirname(path_in_repo),
                recursive=False,
                page_number=page_number,
                page_size=page_size,
            )
            for f in files:
                if f.get('Type') == 'tree':
                    continue
                if f.get('Path') == path_in_repo:
                    return True
            if len(files) < page_size:
                break
            page_number += 1
        return False


class _OSSUploader(Uploader):

    def __init__(self, upload_dict: dict):
        self.__bucket_name, self.__path_prefix = upload_dict['path'].split('/', maxsplit=1)
        self.__path_prefix = self.__path_prefix.removesuffix('/')
        self.__endpoint = 'https://' + upload_dict['url']
        self.__access_key_id = upload_dict['user']
        self.__access_key_secret = upload_dict['password']

    def __make_client(self):
        import alibabacloud_oss_v2 as oss
        cfg = oss.config.load_default()
        cfg.credentials_provider = oss.credentials.StaticCredentialsProvider(self.__access_key_id, self.__access_key_secret)
        cfg.endpoint = self.__endpoint
        cfg.region = re.match(r'^(?:https?://)?oss-([a-z0-9-]+?)(?:-[a-z]+)?(?:-internal)?\.aliyuncs\.com$', self.__endpoint).group(1)
        cfg.retry_max_attempts = 64
        return oss.Client(cfg)

    def __object_key(self, path_in_repo: str) -> str:
        return self.__path_prefix + '/' + path_in_repo

    def exists(self, path_in_repo: str) -> bool:
        import alibabacloud_oss_v2 as oss
        client = self.__make_client()
        try:
            client.head_object(oss.HeadObjectRequest(bucket=self.__bucket_name, key=self.__object_key(path_in_repo)))
            return True
        except oss.exceptions.OperationError as e:
            inner = e.unwrap()
            if isinstance(inner, oss.exceptions.ServiceError) and inner.status_code == 404:
                return False
            raise

    @staticmethod
    def __upload_part(part_number: int, start: int, size: int, filepath: str, client, bucket_name: str, object_key: str, oss_upload_id: str):
        import alibabacloud_oss_v2 as oss
        with open(filepath, 'rb') as f:
            f.seek(start)
            body = f.read(size)
        md5_hash = hashlib.md5()
        md5_hash.update(body)
        result = client.upload_part(oss.UploadPartRequest(
            bucket=bucket_name,
            key=object_key,
            upload_id=oss_upload_id,
            part_number=part_number,
            body=body,
            content_md5=str(base64.b64encode(md5_hash.digest()), 'ascii'),
            content_length=size,
        ))
        return oss.UploadPart(part_number=part_number, etag=result.etag)

    def upload(self, filepath: str, path_in_repo: str) -> None:
        import alibabacloud_oss_v2 as oss
        from alibabacloud_oss_v2.crc import Crc64
        client = self.__make_client()
        object_key = self.__object_key(path_in_repo)

        file_size = os.path.getsize(filepath)
        part_size = int(os.getenv('OSS_PART_SIZE', str(16 * 1024 * 1024)))
        with open(filepath, 'rb') as f:
            crc64 = Crc64(0)
            while True:
                chunk = f.read(8 * 1024 * 1024)
                if not chunk:
                    break
                crc64.update(chunk)
            local_crc64 = str(crc64.sum64())
        oss_upload_id = client.initiate_multipart_upload(oss.InitiateMultipartUploadRequest(bucket=self.__bucket_name, key=object_key)).upload_id

        part_ranges: list[tuple[int, int, int]] = []
        part_number, start = 1, 0
        while start < file_size:
            end = min(start + part_size, file_size)
            part_ranges.append((part_number, start, end - start))
            part_number, start = part_number + 1, end

        try:
            with ThreadPoolExecutor(max_workers=8) as executor:
                upload_parts = [
                    future.result() for future in as_completed([
                        executor.submit(
                            self.__upload_part,
                            part_number=pn,
                            start=st,
                            size=sz,
                            filepath=filepath,
                            client=client,
                            bucket_name=self.__bucket_name,
                            object_key=object_key,
                            oss_upload_id=oss_upload_id,
                        ) for pn, st, sz in part_ranges
                    ])
                ]
        except Exception:
            client.abort_multipart_upload(oss.AbortMultipartUploadRequest(bucket=self.__bucket_name, key=object_key, upload_id=oss_upload_id))
            raise

        upload_parts.sort(key=lambda p: p.part_number)
        client.complete_multipart_upload(oss.CompleteMultipartUploadRequest(
            bucket=self.__bucket_name,
            key=object_key,
            upload_id=oss_upload_id,
            complete_multipart_upload=oss.CompleteMultipartUpload(parts=upload_parts),
        ))
        head_result = client.head_object(oss.HeadObjectRequest(bucket=self.__bucket_name, key=object_key))
        assert head_result.content_length == file_size, f'File size mismatch after upload: local={file_size}, remote={head_result.content_length}'
        assert head_result.hash_crc64 == local_crc64, f'CRC64 mismatch after upload: local={local_crc64}, remote={head_result.hash_crc64}'


class _FileUploader(Uploader):

    def __init__(self, upload_dict: dict):
        assert upload_dict.get('user', None) is None and upload_dict.get('password', None) is None
        self.__base_path = os.path.join(
            {
                '': '/',
                '.': os.getcwd()
            }[upload_dict.get('url', '')],
            upload_dict['path'],
        )

    def __full_path(self, path_in_repo: str) -> str:
        return os.path.join(self.__base_path, path_in_repo)

    def exists(self, path_in_repo: str) -> bool:
        return os.path.exists(self.__full_path(path_in_repo))

    def upload(self, filepath: str, path_in_repo: str) -> None:
        copy_path = self.__full_path(path_in_repo)
        os.makedirs(os.path.dirname(copy_path), exist_ok=True)
        shutil.copy(filepath, copy_path)
