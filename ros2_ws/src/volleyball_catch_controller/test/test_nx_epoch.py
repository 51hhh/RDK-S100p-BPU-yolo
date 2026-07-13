import tempfile
import unittest
from pathlib import Path

from volleyball_catch_controller.nx_epoch import write_epoch


class NxEpochTest(unittest.TestCase):
    def test_writes_nonzero_epoch_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run' / 'nx_source_epoch'
            first = write_epoch(path)
            second = write_epoch(path)
            self.assertGreater(first, 0)
            self.assertGreater(second, 0)
            self.assertEqual(int(path.read_text(encoding='utf-8')), second)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)


if __name__ == '__main__':
    unittest.main()
