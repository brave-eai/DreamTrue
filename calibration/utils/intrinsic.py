from functools import lru_cache
from typing import Any, Protocol

import numpy as np


class OpenCVIntrinsic:
    def __init__(self, fx: float, fy: float, cx: float, cy: float, w: int | None = None, h: int | None = None, k1: float = 0, k2: float = 0, p1: float = 0, p2: float = 0, k3: float = 0):
        self._fx = float(fx)
        self._fy = float(fy)
        self._cx = float(cx)
        self._cy = float(cy)
        self._w = int(w) if w is not None else None
        self._h = int(h) if h is not None else None
        self._k1 = float(k1)
        self._k2 = float(k2)
        self._p1 = float(p1)
        self._p2 = float(p2)
        self._k3 = float(k3)

    def clone(self):
        return self.__class__(fx=self._fx, fy=self._fy, cx=self._cx, cy=self._cy, w=self._w, h=self._h, k1=self._k1, k2=self._k2, p1=self._p1, p2=self._p2, k3=self._k3)

    @property
    def fx(self) -> float:
        return self._fx

    @property
    def fy(self) -> float:
        return self._fy

    @property
    def cx(self) -> float:
        return self._cx

    @property
    def cy(self) -> float:
        return self._cy

    @property
    def w(self) -> int:
        assert self._w is not None, "Width is not specified"
        return self._w

    @property
    def h(self) -> int:
        assert self._h is not None, "Height is not specified"
        return self._h

    @property
    def k1(self) -> float:
        return self._k1

    @property
    def k2(self) -> float:
        return self._k2

    @property
    def p1(self) -> float:
        return self._p1

    @property
    def p2(self) -> float:
        return self._p2

    @property
    def k3(self) -> float:
        return self._k3

    @property
    def fovx(self) -> float:
        return 2 * np.arctan(self.w / (2 * self.fx))

    @property
    def fovy(self) -> float:
        return 2 * np.arctan(self.h / (2 * self.fy))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(fx={self._fx}, fy={self._fy}, cx={self._cx}, cy={self._cy}, w={self._w}, h={self._h}, k1={self._k1}, k2={self._k2}, p1={self._p1}, p2={self._p2}, k3={self._k3})'

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash((self._fx, self._fy, self._cx, self._cy, self._w, self._h, self._k1, self._k2, self._p1, self._p2, self._k3))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, OpenCVIntrinsic):
            return NotImplemented
        return (self._fx == other._fx and self._fy == other._fy and self._cx == other._cx and self._cy == other._cy and self._w == other._w and self._h == other._h and self._k1 == other._k1 and self._k2 == other._k2 and self._p1 == other._p1 and self._p2 == other._p2 and self._k3 == other._k3)

    @classmethod
    def from_dict(cls, data: dict):
        data = data.get('intrinsic', data)
        return cls(fx=data['fx'], fy=data['fy'], cx=data['cx'], cy=data['cy'], w=data.get('w', None), h=data.get('h', None), k1=data.get('k1', 0), k2=data.get('k2', 0), p1=data.get('p1', 0), p2=data.get('p2', 0), k3=data.get('k3', 0))

    @property
    def dict(self) -> dict[str, float|int|None]:
        return {'fx': self._fx, 'fy': self._fy, 'cx': self._cx, 'cy': self._cy, 'k1': self._k1, 'k2': self._k2, 'p1': self._p1, 'p2': self._p2, 'k3': self._k3, 'w': self._w, 'h': self._h}


class OpenCVRenderIntrinsic(OpenCVIntrinsic):
    def __init__(self, fx: float, fy: float, cx: float, cy: float, rfx: float, rfy: float, rcx: float, rcy: float, w: int | None = None, h: int | None = None, k1: float = 0, k2: float = 0, p1: float = 0, p2: float = 0, k3: float = 0):
        super().__init__(fx=fx, fy=fy, cx=cx, cy=cy, w=w, h=h, k1=k1, k2=k2, p1=p1, p2=p2, k3=k3)
        self._rfx = float(rfx)
        self._rfy = float(rfy)
        self._rcx = float(rcx)
        self._rcy = float(rcy)

    def clone(self):
        return self.__class__(fx=self._fx, fy=self._fy, cx=self._cx, cy=self._cy, w=self._w, h=self._h, k1=self._k1, k2=self._k2, p1=self._p1, p2=self._p2, k3=self._k3, rfx=self._rfx, rfy=self._rfy, rcx=self._rcx, rcy=self._rcy)

    @property
    def rfx(self) -> float:
        return self._rfx

    @property
    def rfy(self) -> float:
        return self._rfy

    @property
    def rcx(self) -> float:
        return self._rcx

    @property
    def rcy(self) -> float:
        return self._rcy

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(fx={self._fx}, fy={self._fy}, cx={self._cx}, cy={self._cy}, w={self._w}, h={self._h}, k1={self._k1}, k2={self._k2}, p1={self._p1}, p2={self._p2}, k3={self._k3}, rfx={self._rfx}, rfy={self._rfy}, rcx={self._rcx}, rcy={self._rcy})'

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash((self._fx, self._fy, self._cx, self._cy, self._w, self._h, self._k1, self._k2, self._p1, self._p2, self._k3, self._rfx, self._rfy, self._rcx, self._rcy))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, OpenCVRenderIntrinsic):
            return NotImplemented
        return (self._fx == other._fx and self._fy == other._fy and self._cx == other._cx and self._cy == other._cy and self._w == other._w and self._h == other._h and self._k1 == other._k1 and self._k2 == other._k2 and self._p1 == other._p1 and self._p2 == other._p2 and self._k3 == other._k3 and self._rfx == other._rfx and self._rfy == other._rfy and self._rcx == other._rcx and self._rcy == other._rcy)

    @classmethod
    def from_dict(cls, data: dict):
        data = data.get('intrinsic', data)
        return cls(fx=data['fx'], fy=data['fy'], cx=data['cx'], cy=data['cy'], w=data.get('w', None), h=data.get('h', None), k1=data.get('k1', 0), k2=data.get('k2', 0), p1=data.get('p1', 0), p2=data.get('p2', 0), k3=data.get('k3', 0), rfx=data['rfx'], rfy=data['rfy'], rcx=data['rcx'], rcy=data['rcy'])

    @staticmethod   
    @lru_cache(maxsize=1024)
    def __from_opencv(intr: OpenCVIntrinsic) -> tuple[float, float, float, float, float, float, float, float, int, int, float, float, float, float, float]:
        w, h = intr.w, intr.h
        v_out, u_out = np.mgrid[0:h, 0:w].astype(np.float64) + 0.5
        x_d, y_d = camera_unpinhole(x=u_out,y= v_out,fx= intr.fx, fy=intr.fy, cx=intr.cx,cy= intr.cy)
        x, y = camera_undistort(x=x_d, y=y_d, k1=intr.k1, k2=intr.k2, p1=intr.p1, p2=intr.p2, k3=intr.k3)
        x_min, x_max = float(x.min()), float(x.max())
        y_min, y_max = float(y.min()), float(y.max())
        rfx = (w - 1) / (x_max - x_min)
        rcx = 0.5 - rfx * x_min
        rfy = (h - 1) / (y_max - y_min)
        rcy = 0.5 - rfy * y_min
        return intr.fx, intr.fy, intr.cx, intr.cy, rfx, rfy, rcx, rcy, w, h, intr.k1, intr.k2, intr.p1, intr.p2, intr.k3

    @classmethod
    def from_opencv(cls, intr: OpenCVIntrinsic) -> 'OpenCVRenderIntrinsic':
        fx, fy, cx, cy, rfx, rfy, rcx, rcy, w, h, k1, k2, p1, p2, k3 = cls.__from_opencv(intr)
        return cls(fx=fx, fy=fy, cx=cx, cy=cy, rfx=rfx, rfy=rfy, rcx=rcx, rcy=rcy, w=w, h=h, k1=k1, k2=k2, p1=p1, p2=p2, k3=k3)

    @property
    def dict(self) -> dict[str, float|int|None]:
        return {'fx': self._fx, 'fy': self._fy, 'cx': self._cx, 'cy': self._cy, 'k1': self._k1, 'k2': self._k2, 'p1': self._p1, 'p2': self._p2, 'k3': self._k3, 'w': self._w, 'h': self._h, 'rfx': self._rfx, 'rfy': self._rfy, 'rcx': self._rcx, 'rcy': self._rcy}

    @property
    def opencv(self) -> OpenCVIntrinsic:
        return OpenCVIntrinsic(fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy, w=self.w, h=self.h, k1=self.k1, k2=self.k2, p1=self.p1, p2=self.p2, k3=self.k3)

class _cod(Protocol):
    def __add__(self, other: Any, /) -> Any: ...
    def __radd__(self, other: Any, /) -> Any: ...
    def __sub__(self, other: Any, /) -> Any: ...
    def __rsub__(self, other: Any, /) -> Any: ...
    def __mul__(self, other: Any, /) -> Any: ...
    def __rmul__(self, other: Any, /) -> Any: ...
    def __truediv__(self, other: Any, /) -> Any: ...
    def __rtruediv__(self, other: Any, /) -> Any: ...
    def __pow__(self, exponent: Any, /) -> Any: ...


def camera_distort[T: _cod, V: _cod](x: T, y: T, k1: V, k2: V, p1: V, p2: V, k3: V) -> tuple[T, T]:
    r2 = x ** 2 + y ** 2
    r4, r6 = r2 ** 2, r2 ** 3
    ra = 1 + k1 * r2 + k2 * r4 + k3 * r6
    x_d = x * ra + 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
    y_d = y * ra + p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y
    return x_d, y_d


def camera_undistort[T: _cod, V: _cod](x: T, y: T, k1: V, k2: V, p1: V, p2: V, k3: V, it: int = 8) -> tuple[T, T]:
    x_u, y_u = x, y
    for _ in range(it):
        x_d_guess, y_d_guess = camera_distort(x_u, y_u, k1, k2, p1, p2, k3)
        x_u = x - (x_d_guess - x_u)
        y_u = y - (y_d_guess - y_u)
    return x_u, y_u


def camera_pinhole[T: _cod, V: _cod](x: T, y: T, fx: V, fy: V, cx: V, cy: V) -> tuple[T, T]:
    x = cx + fx * x
    y = cy + fy * y
    return x, y


def camera_unpinhole[T: _cod, V: _cod](x: T, y: T, fx: V, fy: V, cx: V, cy: V) -> tuple[T, T]:
    x = (x - cx) / fx
    y = (y - cy) / fy
    return x, y


def camera_distort_remap[V: _cod](fx: V, fy: V, cx: V, cy: V, k1: V, k2: V, p1: V, p2: V, k3: V, w: int, h: int, rfx: V | None = None, rfy: V | None = None, rcx: V | None = None, rcy: V | None = None, norm: bool = False) -> np.ndarray:
    v_out, u_out = np.mgrid[0:h, 0:w].astype(np.float64) + 0.5
    rfx, rfy, rcx, rcy = (rfx if rfx is not None else fx), (rfy if rfy is not None else fy), (rcx if rcx is not None else cx), (rcy if rcy is not None else cy)
    x_d, y_d = camera_unpinhole(x=u_out, y=v_out, fx=fx, fy=fy, cx=cx, cy=cy)
    x, y = camera_undistort(x_d, y_d, k1=k1, k2=k2, p1=p1, p2=p2, k3=k3)
    u_in, v_in = camera_pinhole(x=x, y=y, fx=rfx, fy=rfy, cx=rcx, cy=rcy)
    if norm:
        u_in = 2.0 * u_in / w - 1.0
        v_in = 2.0 * v_in / h - 1.0
    return np.stack((u_in, v_in), axis=-1)


def get_checkerboard_image(w: int, h: int, x: int = 32, y: int = 32) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    return ((xx // x) % 2) ^ ((yy // y) % 2)
