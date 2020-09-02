import io
import os
from typing import List, Optional

import numpy as np
import oss2
from PIL import Image


class OssFileSystem(object):
    def __init__(self, bucket):
        """
        @param bucket: type=oss2.Bucket
        """
        super(OssFileSystem, self).__init__()
        self.bucket = bucket

    def listdir(self, path: str) -> List[str]:
        """
        @param path: type=str
        @return: type=List[str], element is name of file/sub-directory
        """
        if path[-1] != '/':
            path = path + '/'
        iterator = oss2.ObjectIterator(self.bucket, prefix=path, delimiter='/')
        path_list = [os.path.basename(obj.key.rstrip('/')) for obj in iterator]
        path_list = path_list[1:] # not include directory itself
        return path_list

    def read(self, path: str):
        return self.bucket.get_object(path).read()

    def open(self, path: str):
        return self.bucket.get_object(path)

    def get_bytes_io(self, path: str):
        bytes_io = io.BytesIO(self.read(path))
        bytes_io.seek(0)
        return bytes_io

    def write(self, path: str, buffer: io.BytesIO):
        """
        @param path:
        @param buffer: type=BytesIO
        @return:
        """
        self.bucket.put_object(path, buffer)

    def read_pil_image(self, path: str):
        return Image.open(self.get_bytes_io(path))


    def read_opencv_image(self, path: str):
        """
        @return: shape=(H, W, C), format=BGR
        """
        import cv2
        arr = np.frombuffer(self.read(path), np.uint8)
        image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return image_bgr

    def download(self, remote_path: str, local_path: Optional[str]=None):
        if local_path is None:
            local_path = os.path.basename(remote_path)
        self.bucket.get_object_to_file(remote_path, local_path)


    def put_objects_from_dir(self, dir: str):
        for path, subdirs, files in os.walk(dir):
            for name in files:
                file_path = os.path.join(path, name)
                self.bucket.put_object_from_file(file_path, file_path)


def build_oss_fs(oss_access: dict):
    auth = oss2.Auth(oss_access["access_id"], oss_access["access_key"])
    bucket = oss2.Bucket(auth, oss_access["endpoint"], oss_access["bucket_name"])
    return OssFileSystem(bucket)
