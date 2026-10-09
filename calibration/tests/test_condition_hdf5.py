import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


@unittest.skipUnless(h5py is not None, 'h5py is not installed')
class ConditionHDF5SmokeTest(unittest.TestCase):
    """Build a minimal file following the layout written by calibrate/condition.py
    and check that the bundled `scripts/print-hdf5.py` can inspect it."""

    T, H, W = 2, 8, 10

    def _build_minimal(self, path: pathlib.Path) -> None:
        with h5py.File(path, 'w') as db:
            db.attrs['data_key'] = '000/000000'
            db.attrs['calib_key'] = '000/000000/optimize'
            db.attrs['calib_hash'] = '0' * 64
            for cam in ('head', 'hand_left'):
                db.create_dataset(f'{cam}/rgb', shape=(self.T, self.H, self.W, 3), dtype='uint8')
                db.create_dataset(f'{cam}/mask', shape=(self.T, self.H, self.W), dtype='bool')
                db.create_dataset(f'{cam}/depth', shape=(self.T, self.H, self.W), dtype='uint16')
                db.create_dataset(f'{cam}/segmentation', shape=(self.T, self.H, self.W), dtype='uint16')
                db.create_dataset(f'{cam}/global_pose', shape=(self.T, 4, 4), dtype='float32')
            db.create_dataset('qpos', shape=(self.T, 20), dtype='float32')
            db.create_dataset('position', shape=(self.T, 7), dtype='float32')
            db.create_dataset('indices', data=list(range(self.T)), dtype='int32')

    def test_print_hdf5_reports_cameras_datasets_and_attrs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'condition-example.h5'
            self._build_minimal(path)
            result = subprocess.run(
                [sys.executable, str(ROOT / 'scripts' / 'print-hdf5.py'), str(path)],
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIn('head/', result.stdout)
            self.assertIn('hand_left/', result.stdout)
            self.assertIn('rgb', result.stdout)
            self.assertIn('@data_key', result.stdout)
            self.assertIn('@calib_key', result.stdout)


if __name__ == '__main__':
    unittest.main()
