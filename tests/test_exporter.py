import tempfile
import unittest
from pathlib import Path

from grid_env import Exporter, GridEnv, Importer


class ExporterTests(unittest.TestCase):
    def test_exports_grid_state_as_csv_metadata(self):
        env = GridEnv([3, 2])
        env.start = [2, 1]
        env.path_length = 4
        env.grid = [[0, 1, 2], [3, 4, 5]]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.csv"
            Exporter().call(env, path_to_file=str(path))

            self.assertEqual(
                path.read_text(),
                "# start: 2,1\n"
                "# path_length: 4\n"
                "0,1,2\n"
                "3,4,5\n",
            )
            imported = Importer().call(path_to_file=str(path))

            self.assertEqual(imported.grid, env.grid)
            self.assertEqual(imported.start, env.start)
            self.assertEqual(imported.path_length, env.path_length)

    def test_imports_grid_without_metadata_using_default_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.csv"
            path.write_text("0,1\n2,3\n")

            imported = Importer().call(path_to_file=str(path))

        self.assertEqual(imported.grid, [[0, 1], [2, 3]])
        self.assertEqual(imported.start, [0, 0])
        self.assertEqual(imported.path_length, 0)


if __name__ == "__main__":
    unittest.main()
