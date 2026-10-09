from typing import TYPE_CHECKING, Literal

import numpy as np

try:
    import sapien as _sapien_module
except ImportError:
    _sapien_module = None  # type: ignore[assignment]

try:
    import mplib as _mplib_module
except ImportError:
    _mplib_module = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import sapien
    import mplib

from quaternion import as_euler_angles, as_float_array, as_rotation_matrix, from_euler_angles, from_float_array, from_rotation_matrix, quaternion


class Pose:

    def __init__(self, p: None | np.ndarray | tuple[float, float, float] = None, q: None | np.ndarray | tuple[float, float, float, float] = None):
        if p is None:
            p = np.zeros((3, ), dtype=np.float64)
        if q is None:
            q = np.array([1., 0., 0., 0.], dtype=np.float64)
        if isinstance(p, tuple | list):
            p = np.array(p, dtype=np.float64)
        if isinstance(q, tuple | list):
            q = np.array(q, dtype=np.float64)
        p = p.squeeze()
        q = q.squeeze()
        if isinstance(q, quaternion):
            q = as_float_array(q)
        assert p.shape == (3, ), f"Invalid translation shape: {p.shape}"
        self.__p: np.ndarray[Literal[3], np.dtype[np.floating]] = p  # x y z
        assert q.shape == (4, ), f"Invalid qernion shape: {q.shape}"
        q = (-q) if q[0] < 0 else q
        self.__q: np.ndarray[Literal[4], np.dtype[np.floating]] = q  # w x y z

    def clone(self) -> 'Pose':
        return self.__class__(self.__p.copy(), self.__q.copy())

    def __hash__(self):
        return hash((self.__p.tobytes(), self.__q.tobytes()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pose):
            return NotImplemented
        return np.array_equal(self.__p, other.__p) and np.array_equal(self.__q, other.__q)

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.__p}, q={self.__q})'

    def __str__(self):
        return self.__repr__()

    @classmethod
    def from_dict(cls, data: dict):
        return cls(p=data['p'], q=data['q'])

    @classmethod
    def from_list(cls, data: list[float] | np.ndarray[Literal[7], np.dtype[np.floating]]):
        assert len(data) == 7, f"Invalid data length: {len(data)}"
        data = np.asarray(data, dtype=np.float64)
        assert not np.allclose(data, 0), "Zero pose is not allowed."
        return cls(data[:3], data[3:])

    @classmethod
    def from_transformation(cls, t: np.ndarray[Literal[4, 4], np.dtype[np.floating]]):
        assert t.shape == (4, 4), f"Invalid transformation shape: {t.shape}"
        return cls(t[:3, 3], as_float_array(from_rotation_matrix(t[:3, :3])))

    @property
    def translation(self) -> np.ndarray[Literal[3], np.dtype[np.floating]]:
        return self.__p

    @property
    def quaternion(self) -> np.ndarray[Literal[4], np.dtype[np.floating]]:
        return self.__q

    @property
    def p(self) -> np.ndarray[Literal[3], np.dtype[np.floating]]:
        return self.__p

    @property
    def q(self) -> np.ndarray[Literal[4], np.dtype[np.floating]]:
        return self.__q

    @property
    def rotation(self) -> np.ndarray:
        return self.quat_to_rotation(self.__q)

    @property
    def r6d(self) -> np.ndarray:
        return self.quat_to_6d(self.__q)

    @property
    def rpy(self) -> np.ndarray:
        return self.quat_to_rpy(self.__q)

    @property
    def transformation(self) -> np.ndarray[Literal[4, 4], np.dtype[np.floating]]:
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation
        return T

    def __matmul__(self, other: 'Pose') -> 'Pose':
        if not isinstance(other, Pose):
            raise TypeError(f"Unsupported operand type(s) for @: '{self.__class__.__name__}' and '{other.__class__.__name__}'")
        return self.__class__.from_transformation(self.transformation @ other.transformation)

    @property
    def inv(self) -> 'Pose':
        return self.__class__.from_transformation(np.linalg.inv(self.transformation))

    @classmethod
    def random(cls, x: float, y: float, z: float, pitch: float, roll: float, yaw: float):
        xyz = np.array([x, y, z])
        rpy = np.deg2rad(np.array([pitch, roll, yaw]))
        return cls(p=np.random.uniform(-xyz, xyz), q=from_euler_angles(np.random.uniform(-rpy, rpy)))

    def __sub__(self, other):
        if not isinstance(other, Pose):
            raise TypeError(f"Unsupported operand type(s) for -: '{self.__class__.__name__}' and '{other.__class__.__name__}'")
        p_dist = np.linalg.norm(self.translation - other.translation)
        q1 = self.q / np.linalg.norm(self.q)
        q2 = other.q / np.linalg.norm(other.q)
        dot = np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0)
        rot_angle = 2.0 * np.arccos(dot)
        return p_dist, rot_angle

    @staticmethod
    def quat_to_6d(quad: np.ndarray | list[float]) -> np.ndarray:
        return as_rotation_matrix(from_float_array(np.array(quad)))[:, :2]

    @staticmethod
    def quat_to_rotation(quad: np.ndarray | list[float]) -> np.ndarray:
        return as_rotation_matrix(from_float_array(np.array(quad)))

    @staticmethod
    def quat_to_rpy(quad: np.ndarray | list[float]) -> np.ndarray:
        return as_euler_angles(from_float_array(np.array(quad)))

    @classmethod
    def from_sapien(cls, pose: 'sapien.Pose') -> 'Pose':
        if _sapien_module is None:
            raise ImportError("sapien is not installed")
        return cls(p=np.array(pose.p), q=np.array(pose.q))

    @property
    def sapien(self) -> 'sapien.Pose':
        if _sapien_module is None:
            raise ImportError("sapien is not installed")
        return _sapien_module.Pose(p=self.__p, q=self.__q)

    @classmethod
    def from_mplib(cls, pose: 'mplib.Pose') -> 'Pose':
        if _mplib_module is None:
            raise ImportError("mplib is not installed")
        return cls(p=np.array(pose.p), q=np.array(pose.q))

    @property
    def mplib(self) -> 'mplib.Pose':
        if _mplib_module is None:
            raise ImportError("mplib is not installed")
        return _mplib_module.Pose(p=self.__p, q=self.__q)

    @property
    def dict(self) -> dict:
        return {
            "p": self.__p.tolist(),
            "q": self.__q.tolist(),
        }

    @property
    def list(self) -> list[float]:
        return self.__p.tolist() + self.__q.tolist()


class Extrinsic(Pose):
    # isaac sim: +Z上 +X前 +Y左
    # opencv   : +Y下 +Z前 +X右
    # sapien   : +Z上 +X前
    # OpenGL   : +Y上 -Z前
    # Blender  : +Y上 -Z前
    ...
