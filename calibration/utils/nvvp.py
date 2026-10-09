import subprocess
import traceback

import PyNvVideoCodec as nvc
import cv2
import torch


def video_read(video_path: str, gpu_id: int = 0) -> list[torch.Tensor]:
    nv_dec = nvc.SimpleDecoder(
        enc_file_path=video_path,
        gpu_id=gpu_id,
        output_color_type=nvc.OutputColorType.RGB,
    )
    images = []
    for _ in range(0, len(nv_dec), 16):
        for frame in nv_dec.get_batch_frames(16):
            tensor = torch.from_dlpack(frame)
            print(tensor.shape, tensor.device, tensor.dtype)
            images.append(tensor.clone())
    cv2.imwrite('decoded_image.jpg', images[0].cpu().numpy())
    return images


class NvEncoder:
    def __init__(
            self,
            file_name: str,
            fps: int = 30,
            gpu_id: int = 0,
            codec: str = 'hevc',
            width: int = 640,
            height: int = 480,
            fmt: str = 'YUV444',
    ):
        self.__width = width
        self.__height = height
        self.__enc = nvc.CreateEncoder(
            gpuid=gpu_id,
            codec=codec,
            width=width,
            height=height,
            bitrate=10000000,
            rc='constqp',
            constqp=0,  # important for lossless encoding
            bf=0,
            fps=fps,
            fmt=fmt,
            usecpuinputbuffer=False
        )
        self.__fmt = fmt
        self.__proc = subprocess.Popen(['ffmpeg', '-f', codec, '-r', str(fps), '-i', 'pipe:0', '-vcodec', 'copy', '-y', file_name], stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def __write(self, chunk: bytearray):
        if self.__proc.poll() is None:
            self.__proc.stdin.write(chunk)
            self.__proc.stdin.flush()
        else:
            raise RuntimeError("FFmpeg process has terminated.")

    def close(self):
        self.__write(bytearray(self.__enc.EndEncode()))
        if self.__proc.stdin:
            self.__proc.stdin.close()
        self.__proc.wait()

    def __call__(self, frame: torch.Tensor):
        """frame should be in RGB format and of shape (height, width, 3) dtype=float in [0,1] or dtype=uint8 in [0,255]"""
        assert frame.shape == (self.__height, self.__width, 3)
        frame = {
            'YUV444': self.__rgb_to_yuv444,
            'ABGR': self.__rgb_to_agbr,
        }[self.__fmt](frame)
        bitstream = self.__enc.Encode(frame)
        self.__write(bytearray(bitstream))

    @staticmethod
    def __rgb_to_yuv444(frame: torch.Tensor) -> torch.Tensor:
        if frame.dtype == torch.uint8:
            frame = frame.float() / 255.0
        h, w, c = frame.shape
        r, g, b = frame.unbind(dim=-1)
        y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
        u: torch.Tensor = ((-0.147 * r - 0.289 * g + 0.436 * b)  + 1) * 0.5
        v: torch.Tensor = ((0.615 * r - 0.515 * g - 0.100 * b) + 1) * 0.5
        out: torch.Tensor = torch.stack([y, u, v], dim=-3)
        out = (out * 255).clip(0, 255).to(torch.uint8).contiguous().view(h * 3, w).contiguous()
        return out

    @staticmethod
    def __rgb_to_agbr(frame: torch.Tensor) -> torch.Tensor:
        if frame.dtype == torch.float:
            frame = (frame * 255).clip(0, 255).to(torch.uint8)
        return torch.cat([
            frame,
            torch.ones(frame.shape[:-1] + (1,), dtype=frame.dtype, device=frame.device) * 255
        ], dim=-1).contiguous()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class MultyNvEncoder[T]:
    def __init__(
            self,
            file_name: dict[T, str],
            gpu_id: int = 0,
            codec: str = 'hevc',
            width: int = 640,
            height: int = 480,
            fps: int = 30,
            fmt: str = 'YUV444',
    ):
        self.__encoders: dict[T, NvEncoder] = {name: NvEncoder(file_name=file, gpu_id=gpu_id, codec=codec, width=width, height=height, fps=fps, fmt=fmt) for name, file in file_name.items()}

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        exc = None
        for encoder in self.__encoders.values():
            try:
                encoder.close()
            except Exception as e:
                traceback.print_exc()
                exc = e
        if exc is not None:
            raise exc

    def __call__(self, frame: dict[T, torch.Tensor]):
        for name, encoder in self.__encoders.items():
            encoder(frame[name])
