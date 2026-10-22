import unittest
from unittest.mock import Mock

import numpy as np
from gymnasium import spaces

from gym_grid_env import GymGridEnv


class GymGridEnvTests(unittest.TestCase):
    def setUp(self):
        self.grid = Mock()
        self.grid.ACTIONS = {
            "top": [0, -1],
            "bottom": [0, 1],
            "left": [-1, 0],
            "right": [1, 0],
        }
        self.grid.TYPES = {
            "empty": {"id": 0},
            "finish": {"id": 1},
            "bonus": {"id": 2, "count": 2},
            "wall": {"id": 3},
        }
        self.grid.width = 3
        self.grid.height = 2
        self.grid.start = [0, 0]
        self.grid.grid = [
            [0, 2, 1],
            [3, 0, 0],
        ]
        self.grid.get_state.side_effect = lambda pos, size: [
            [0] * size[0] for _ in range(size[1])
        ]

        self.agent = Mock()
        self.agent.window_size = [3, 3]
        self.agent.TENSOR_LAYERS_COUNT = 6
        self.agent.get_observation_tensor.return_value = np.zeros(
            (6, 3, 3), dtype=np.int32
        )
        self.agent.consume_bonus.return_value = True

        self.env = GymGridEnv(self.grid, self.agent, max_steps=2)

    def test_initializes_gym_spaces_and_copies_start_position(self):
        self.assertIsInstance(self.env.action_space, spaces.Discrete)
        self.assertEqual(self.env.action_space.n, 4)
        self.assertEqual(self.env.action_list, ["top", "bottom", "left", "right"])
        self.assertIsInstance(self.env.observation_space, spaces.Box)
        self.assertEqual(self.env.observation_space.shape, (6, 3, 3))
        self.assertEqual(self.env.observation_space.dtype, np.dtype(np.int32))
        self.assertEqual(self.env.agent_pos, [0, 0])
        self.assertEqual(self.env.agent_pos, self.grid.start)

    def test_reset_returns_observation_and_info_and_resets_episode_counters(self):
        self.env.current_step = 7
        self.env.current_episode_reward = 25
        self.env.agent_pos = [1, 1]

        observation, info = self.env.reset(seed=17)

        self.assertEqual(observation.shape, (6, 3, 3))
        self.assertEqual(observation.dtype, np.int32)
        self.assertEqual(info, {"agent_pos": [0, 0]})
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(self.env.current_episode_reward, 0)
        self.grid.get_state.assert_called_with([0, 0], [3, 3])
        self.agent.update_memory.assert_called_with([0, 0], [[0] * 3 for _ in range(3)])

    def test_step_to_empty_cell_updates_position_and_reward(self):
        self.env.agent_pos = [1, 1]

        observation, reward, terminated, truncated, info = self.env.step(3)

        self.assertEqual(reward, self.env.REWARDS["empty"])
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info, {"agent_pos": [2, 1], "cell_type": 0})
        self.assertEqual(observation.shape, (6, 3, 3))

    def test_bonus_is_rewarded_only_when_agent_consumes_it(self):
        observation, reward, terminated, truncated, info = self.env.step(3)

        self.assertEqual(reward, self.env.REWARDS["bonus"])
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info, {"agent_pos": [1, 0], "cell_type": 2})
        self.assertEqual(observation.shape, (6, 3, 3))
        self.assertEqual(self.env.current_step, 1)
        self.assertEqual(self.env.current_episode_reward, 10)
        self.agent.consume_bonus.assert_called_once_with([1, 0])

        self.env.agent_pos = [0, 0]
        self.agent.consume_bonus.return_value = False
        _, reward, terminated, truncated, _ = self.env.step(3)
        self.assertEqual(reward, self.env.REWARDS["empty"])
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(self.env.agent_pos, [1, 0])

    def test_step_to_finish_terminates_and_awards_finish_reward(self):
        self.env.agent_pos = [1, 0]

        _, reward, terminated, truncated, info = self.env.step(3)

        self.assertEqual(reward, self.env.REWARDS["finish"])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["cell_type"], self.grid.TYPES["finish"]["id"])
        self.assertEqual(self.env.agent_pos, [2, 0])

    def test_step_into_wall_penalizes_without_moving(self):
        _, reward, terminated, truncated, info = self.env.step(1)

        self.assertEqual(reward, self.env.REWARDS["wall"])
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(self.env.agent_pos, [0, 0])
        self.assertEqual(info, {"agent_pos": [0, 0], "cell_type": 3})

    def test_step_out_of_bounds_is_a_wall_without_indexing_outside_grid(self):
        _, reward, terminated, truncated, info = self.env.step(0)

        self.assertEqual(reward, self.env.REWARDS["wall"])
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(self.env.agent_pos, [0, 0])
        self.assertEqual(info["cell_type"], self.grid.TYPES["wall"]["id"])

    def test_episode_reward_cannot_exceed_theoretical_maximum(self):
        self.env.current_episode_reward = self.env.max_possible_reward

        with self.assertRaisesRegex(ValueError, "exceeds the absolute theoretical maximum"):
            self.env.step(3)

    def test_max_possible_reward_uses_configured_bonus_count(self):
        self.assertEqual(self.env.max_possible_reward, 120)
        self.grid.TYPES["bonus"]["count"] = 10
        self.assertEqual(self.env.max_possible_reward, 120)


if __name__ == "__main__":
    unittest.main()