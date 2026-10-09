import os
import shutil

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np

from .tempdir import TemporaryDirectory


def video_write(video: np.ndarray | list[np.ndarray], output_path: str, fps: int = 30):
    H, W, C = video[0].shape
    with TemporaryDirectory() as temp_dir:
        temp_video_path = os.path.join(temp_dir, os.path.basename(output_path))
        with av.open(temp_video_path, mode='w') as con:
            stream = con.add_stream('libx264', rate=fps)
            stream.width = W
            stream.height = H
            stream.pix_fmt = 'yuv420p'
            for frame in video:
                assert frame.shape == (H, W, C)
                frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
                for packet in stream.encode(frame):
                    con.mux(packet)
            for packet in stream.encode():
                con.mux(packet)
        shutil.move(temp_video_path, output_path)


def video_read(video_path: str, resolution: None | tuple[int, int] = None) -> np.ndarray:

    def resize(frame: np.ndarray):
        if resolution is not None and frame.shape[:2] != resolution:
            frame = cv2.resize(frame, resolution, interpolation=cv2.INTER_LINEAR)
        return frame

    with TemporaryDirectory() as temp_dir:
        temp_video_path = os.path.join(temp_dir, os.path.basename(video_path))
        shutil.copy(video_path, temp_video_path)
        with av.open(temp_video_path) as con:
            frames = [resize(frame.to_ndarray(format='rgb24')) for frame in con.decode(video=0)]
    return np.stack(frames, axis=0)
