import hashlib
import json
import pathlib
import re
import unittest

FIXTURES = pathlib.Path(__file__).resolve().parent / 'fixtures'
CALIB_EXAMPLE = FIXTURES / 'calibration-example'
HASH_RE = re.compile(r'^[0-9a-f]{64}$')


class TaskFixtureTest(unittest.TestCase):
    """The public task/calibration fixtures must stay consistent with the
    conventions documented in README.md and implemented by
    calibrate/condition.py + sim/SapienEnv.py load_calibration."""

    def test_task_example_has_expected_fields(self):
        tasks = json.loads((FIXTURES / 'task-example.json').read_text(encoding='utf-8'))
        self.assertIsInstance(tasks, list)
        self.assertTrue(tasks)
        for task in tasks:
            self.assertIsInstance(task['data_key'], str)
            self.assertIn('/', task['data_key'])
            self.assertIsInstance(task['calib_key'], str)
            self.assertRegex(task['calib_hash'], HASH_RE)
            self.assertIsInstance(task.get('meta_data', {}), dict)
            self.assertIsInstance(task.get('use_state', False), bool)
            self.assertIsInstance(task.get('modifications', []), list)

    def test_example_calib_hash_matches_calibration_files(self):
        tasks = json.loads((FIXTURES / 'task-example.json').read_text(encoding='utf-8'))
        # sha256 over the concatenated bytes of the calibration files actually loaded,
        # in the order local_pose -> intrinsic -> offset_joints;
        # see utils/sha256.py and sim/SapienEnv.py load_calibration.
        h = hashlib.sha256()
        for name in ('local_pose_best.json', 'intrinsic_best.json'):
            h.update((CALIB_EXAMPLE / name).read_bytes())
        self.assertEqual(tasks[0]['calib_hash'], h.hexdigest())

    def test_calibration_example_files_have_required_fields(self):
        local_pose = json.loads((CALIB_EXAMPLE / 'local_pose_best.json').read_text(encoding='utf-8'))
        intrinsic = json.loads((CALIB_EXAMPLE / 'intrinsic_best.json').read_text(encoding='utf-8'))
        self.assertTrue(local_pose and intrinsic)
        for cam, pose in local_pose.items():
            self.assertEqual(3, len(pose['p']), cam)
            self.assertEqual(4, len(pose['q']), cam)
        for cam, intr in intrinsic.items():
            for field in ('fx', 'fy', 'cx', 'cy', 'w', 'h'):
                self.assertIn(field, intr, cam)


if __name__ == '__main__':
    unittest.main()
