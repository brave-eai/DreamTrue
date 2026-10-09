import hashlib
import pathlib


def sha256(*files):
    sha256_hash = hashlib.sha256()
    for file in files:
        if isinstance(file, str) or isinstance(file, pathlib.Path):
            with open(str(file), 'rb') as f:
                for byte_block in iter(lambda: f.read(65536), b''):
                    sha256_hash.update(byte_block)
        elif isinstance(file, bytes):
            sha256_hash.update(file)
        else:
            raise NotImplementedError(f'Unsupported input type: {type(file)}')
    return sha256_hash.hexdigest()
