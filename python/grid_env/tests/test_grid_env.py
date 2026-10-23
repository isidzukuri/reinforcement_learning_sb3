import itertools
import unittest
from collections import Counter
from unittest.mock import patch

from grid_env import GridEnv


class GridEnvTests(unittest.TestCase):
    def test_initializes_grid_and_state(self):
        env = GridEnv([4, 3])

        self.assertEqual(env.width, 4)
        self.assertEqual(env.height, 3)
        self.assertEqual(env.grid, [[0] * 4 for _ in range(3)])
        self.assertEqual(env.start, [0, 0])
        self.assertEqual(env.path, [])
        self.assertEqual(env.path_length, 0)

    def test_get_state_returns_centered_window(self):
        env = GridEnv([4, 3])
        env.grid = [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
            [2, 3, 0, 1],
        ]

        self.assertEqual(
            env.get_state([1, 1], [3, 3]),
            [[0, 1, 2], [1, 2, 3], [2, 3, 0]],
        )

    def test_get_state_fills_out_of_bounds_with_walls(self):
        env = GridEnv([2, 2])
        env.grid = [[0, 1], [2, 3]]

        self.assertEqual(
            env.get_state([0, 0], [3, 3]),
            [[3, 3, 3], [3, 0, 1], [3, 2, 3]],
        )

    def test_generate_creates_valid_path_and_object_counts(self):
        object_ids = {kind: value["id"] for kind, value in GridEnv.TYPES.items()}

        for orientation in itertools.product([False, True], repeat=3):
            with self.subTest(orientation=orientation):
                env = GridEnv([10, 10])
                with patch(
                    "grid_env.grid_env.random.choice",
                    side_effect=orientation,
                ):
                    env.generate()

                expected_path_length = (env.width * env.height + 1) // 2
                self.assertEqual(env.path_length, expected_path_length)
                self.assertEqual(len(env.path), expected_path_length)
                self.assertEqual(env.path[0], env.start)
                self.assertEqual(len({tuple(cell) for cell in env.path}), len(env.path))

                for first, second in zip(env.path, env.path[1:]):
                    self.assertEqual(
                        abs(first[0] - second[0]) + abs(first[1] - second[1]),
                        1,
                    )

                self.assertEqual(
                    env.grid[env.path[-1][1]][env.path[-1][0]],
                    object_ids["finish"],
                )
                for x, y in env.path[:-1]:
                    self.assertNotEqual(env.grid[y][x], object_ids["wall"])

                counts = Counter(value for row in env.grid for value in row)
                self.assertEqual(counts[object_ids["finish"]], 1)
                self.assertEqual(counts[object_ids["bonus"]], 5)
                self.assertEqual(counts[object_ids["wall"]], 15)
                self.assertEqual(counts[object_ids["empty"]], 79)

    def test_generate_randomizes_start_along_path(self):
        env = GridEnv([10, 10])
        with (
            patch("grid_env.grid_env.random.choice", side_effect=[True, False, False]),
            patch("grid_env.grid_env.random.randrange", return_value=15),
        ):
            env.generate()

        self.assertEqual(env.start, [4, 1])
        self.assertEqual(env.path[0], env.start)
        self.assertEqual(len(env.path), env.path_length)

    def test_generate_builds_path_backwards_for_start_near_traversal_end(self):
        env = GridEnv([10, 10])
        with (
            patch("grid_env.grid_env.random.choice", side_effect=[True, False, False]),
            patch("grid_env.grid_env.random.randrange", return_value=99),
        ):
            env.generate()

        self.assertEqual(env.start, [0, 9])
        self.assertEqual(env.path[0], env.start)
        self.assertEqual(len(env.path), env.path_length)
        for first, second in zip(env.path, env.path[1:]):
            self.assertEqual(
                abs(first[0] - second[0]) + abs(first[1] - second[1]),
                1,
            )

    def test_generate_rejects_non_positive_dimensions(self):
        for size in ([0, 10], [10, 0], [-1, 10], [10, -1]):
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, "dimensions must be positive"):
                    GridEnv(size).generate()

    def test_generate_rejects_grid_too_small_for_path_and_walls(self):
        with self.assertRaisesRegex(ValueError, "too small"):
            GridEnv([5, 5]).generate()


if __name__ == "__main__":
    unittest.main()
