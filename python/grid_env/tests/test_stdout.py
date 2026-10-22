import unittest
from contextlib import redirect_stdout
from io import StringIO

from grid_env import GridEnv
from grid_env.output import Stdout


class StdoutTests(unittest.TestCase):
    def setUp(self):
        self.grid = GridEnv([5, 1])
        self.grid.grid = [[0, 1, 2, 3, 99]]
        self.grid.start = [2, 0]
        self.grid.path_length = 4

    def test_prints_configuration_and_grid(self):
        output = StringIO()

        with redirect_stdout(output):
            Stdout().call(self.grid)

        self.assertEqual(
            output.getvalue(),
            "=== GridEnv Configuration ===\n"
            "Size (Width, Height): (5, 1)\n"
            "Start Coordinate: [2, 0]\n"
            "Actual Path Length: 4\n"
            "TYPES:\n"
            "  - empty: {'id': 0, 'count': None}\n"
            "  - finish: {'id': 1, 'count': 1}\n"
            "  - bonus: {'id': 2, 'count': 5}\n"
            "  - wall: {'id': 3, 'count': 15}\n"
            "ACTIONS: ['top', 'bottom', 'left', 'right']\n"
            "\n=== Grid Matrix ===\n"
            " 0  1  2  3 99\n"
            "============================\n",
        )

    def test_highlights_path_coordinates(self):
        output = StringIO()

        with redirect_stdout(output):
            Stdout().call(self.grid, path=[[2, 0]])

        self.assertIn(
            "\033[48;2;144;238;144m 2\033[0m",
            output.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
