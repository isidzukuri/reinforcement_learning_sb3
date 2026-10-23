import unittest
from unittest.mock import Mock, patch

import numpy as np
from gymnasium import spaces

from gym_grid_env import GymGridEnv


class GymGridEnvTests(unittest.TestCase):
    def setUp(self):
        self.grid_env_class = patch("gym_grid_env.gym_grid_env.GridEnv")
        mocked_grid_env = self.grid_env_class.start()
        self.addCleanup(self.grid_env_class.stop)
        self.mocked_grid_env = mocked_grid_env

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
        self.agent.is_revisit.return_value = False

        mocked_grid_env.ACTIONS = self.grid.ACTIONS
        mocked_grid_env.return_value = self.grid

        self.env = GymGridEnv(self.agent, next_stages=[])
        self.env.set_stage({
            "title": "initial",
            "env_size": [3, 2],
            "max_steps": 2,
            "reward_treshold": 100,
        })

    def test_initializes_gym_spaces_from_agent_and_stage(self):
        self.assertIsInstance(self.env.action_space, spaces.Discrete)
        self.assertEqual(self.env.action_space.n, 4)
        self.assertEqual(self.env.action_list, ["top", "bottom", "left", "right"])
        self.assertIsInstance(self.env.observation_space, spaces.Box)
        self.assertEqual(self.env.observation_space.shape, (6, 3, 3))
        self.assertEqual(self.env.observation_space.dtype, np.dtype(np.int32))
        self.assertEqual(self.env.agent_pos, [0, 0])
        self.assertEqual(self.env.agent_pos, self.grid.start)
        self.mocked_grid_env.assert_called_once_with(size=[3, 2])
        self.assertEqual(self.env.max_steps, 2)

    def test_set_stage_configures_grid_and_resets_agent_memory(self):
        self.mocked_grid_env.reset_mock()
        self.agent.reset.reset_mock()
        self.grid.start = [1, 1]
        stage = {
            "title": "larger grid",
            "env_size": [5, 4],
            "max_steps": 20,
            "reward_treshold": 150,
        }

        self.env.set_stage(stage)

        self.assertIs(self.env.current_stage, stage)
        self.assertEqual(self.env.max_steps, 20)
        self.mocked_grid_env.assert_called_once_with(size=[5, 4])
        self.assertIs(self.env.grid_env, self.grid)
        self.assertEqual(self.env.agent_pos, [1, 1])
        self.assertEqual(self.agent.memory_grid_width, 5)
        self.assertEqual(self.agent.memory_grid_height, 4)
        self.agent.reset.assert_called_once_with()
        self.assertEqual(self.env.episodes_count, 0)
        self.assertEqual(self.env.episodes_successful, 0)

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

    def test_revisit_receives_additional_penalty(self):
        self.agent.is_revisit.return_value = True
        self.agent.consume_bonus.return_value = False

        _, reward, terminated, truncated, _ = self.env.step(3)

        self.assertEqual(
            reward,
            self.env.REWARDS["empty"] + self.env.REWARDS["revisit"],
        )
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.agent.is_revisit.assert_called_once_with([1, 0])

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
        self.assertEqual(self.env.episodes_count, 1)
        self.assertEqual(self.env.episodes_successful, 0)

    def test_step_to_finish_terminates_and_awards_finish_reward(self):
        self.env.agent_pos = [1, 0]
        self.env.recent_successes.extend([1] * 99)

        _, reward, terminated, truncated, info = self.env.step(3)

        self.assertEqual(reward, self.env.REWARDS["finish"])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["cell_type"], self.grid.TYPES["finish"]["id"])
        self.assertEqual(self.env.agent_pos, [2, 0])
        self.assertEqual(self.env.episodes_count, 1)
        self.assertEqual(self.env.episodes_successful, 1)
        self.assertTrue(self.env.all_stages_completed)

    def test_episode_success_depends_on_stage_reward_threshold(self):
        self.env.current_episode_reward = 99
        self.assertFalse(self.env.is_episode_successful())

        self.env.current_episode_reward = 100
        self.assertTrue(self.env.is_episode_successful())

    def test_stage_readiness_requires_one_hundred_recent_successes(self):
        self.env.recent_successes.extend([1] * 99)
        self.assertFalse(self.env.is_ready_for_next_stage())

        self.env.recent_successes.append(1)
        self.assertTrue(self.env.is_ready_for_next_stage())

        self.env.recent_successes[-1] = 0
        self.assertFalse(self.env.is_ready_for_next_stage())

    def test_next_stage_uses_next_queued_stage_and_resets_stage_counters(self):
        next_stage = {
            "title": "next",
            "env_size": [4, 3],
            "max_steps": 12,
            "reward_treshold": 110,
        }
        self.env.next_stages.append(next_stage)
        self.env.episodes_count = 5
        self.env.episodes_successful = 4

        self.env.next_stage()

        self.assertIs(self.env.current_stage, next_stage)
        self.assertEqual(self.env.max_steps, 12)
        self.assertEqual(self.env.next_stages, [])
        self.assertEqual(self.env.episodes_count, 0)
        self.assertEqual(self.env.episodes_successful, 0)
        self.assertFalse(self.env.all_stages_completed)

    def test_next_stage_marks_curriculum_complete_when_queue_is_empty(self):
        self.env.next_stage()

        self.assertTrue(self.env.all_stages_completed)

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

    def test_episode_reward_can_exceed_theoretical_maximum(self):
        self.env.current_episode_reward = self.env.max_possible_reward

        _, reward, _, _, _ = self.env.step(3)

        self.assertEqual(reward, self.env.REWARDS["bonus"])
        self.assertGreater(
            self.env.current_episode_reward,
            self.env.max_possible_reward,
        )

    def test_max_possible_reward_uses_configured_bonus_count(self):
        self.grid.TYPES["bonus"]["count"] = 10
        self.assertEqual(self.env.max_possible_reward, 200)


if __name__ == "__main__":
    unittest.main()