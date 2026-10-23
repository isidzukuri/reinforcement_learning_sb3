import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np

from agent import Agent


class AgentTests(unittest.TestCase):
    TYPE_IDS = {"empty": 0, "finish": 1, "bonus": 2, "wall": 3}


    def test_init_sets_all_instance_variables(self):
        window_size = [4, 3]
        type_ids = {"empty": 0, "finish": 1, "bonus": 2, "wall": 3}

        agent = Agent(window_size=window_size, type_ids=type_ids)

        self.assertEqual(
            set(vars(agent)),
            {"window_size", "type_ids", "model", "known_env_map", "memory_grid_height","memory_grid_width"},
        )
        self.assertIs(agent.window_size, window_size)
        self.assertIs(agent.type_ids, type_ids)
        self.assertIsNone(agent.model)
        self.assertEqual(agent.known_env_map.shape, (6, 15, 15))
        self.assertEqual(agent.known_env_map.dtype, np.int32)
        np.testing.assert_array_equal(
            agent.known_env_map,
            np.zeros((6, 15, 15), dtype=np.int32),
        )

    def test_init_rejects_type_ids_missing_required_types(self):
        with self.assertRaisesRegex(
            ValueError,
            r"type_ids is missing required keys: bonus, empty, finish, wall",
        ):
            Agent(window_size=[3, 3], type_ids={})

    def test_validate_type_ids_accepts_all_required_types(self):
        Agent._validate_type_ids(
            {"empty": 0, "finish": 1, "bonus": 2, "wall": 3}
        )

    def test_validate_type_ids_reports_missing_types_in_sorted_order(self):
        with self.assertRaisesRegex(
            ValueError,
            r"type_ids is missing required keys: bonus, empty, finish, wall",
        ):
            Agent._validate_type_ids({})

    def test_load_model_stores_loaded_model(self):
        agent = Agent(window_size=[3, 3], type_ids=self.TYPE_IDS)
        loaded_model = object()

        with patch("agent.agent.DQN.load", return_value=loaded_model) as load:
            agent.load_model("model.zip")

        load.assert_called_once_with("model.zip")
        self.assertIs(agent.model, loaded_model)

    def test_reset_reinitializes_memory_map(self):
        agent = Agent(
            window_size=[3, 3],
            type_ids=self.TYPE_IDS,
            memory_grid_size=[4, 2],
        )
        previous_map = agent.known_env_map
        agent.known_env_map[0, 0, 0] = 1

        agent.reset()

        self.assertIsNot(agent.known_env_map, previous_map)
        self.assertEqual(agent.known_env_map.shape, (6, 2, 4))
        self.assertEqual(agent.known_env_map.dtype, np.int32)
        self.assertFalse(agent.known_env_map.any())

    def test_observation_maps_known_cells_and_pads_outside_grid_with_walls(self):
        agent = Agent(
            window_size=[3, 3],
            type_ids=self.TYPE_IDS,
            memory_grid_size=[2, 2],
        )
        agent.known_env_map[:, 0, 0] = [1, 2, 3, 4, -1, 6]
        agent.known_env_map[:, 0, 1] = [7, 8, 9, 10, 11, 12]

        observation = agent.get_observation_tensor([0, 0])

        self.assertEqual(observation.shape, (6, 3, 3))
        self.assertEqual(observation.dtype, np.int32)
        np.testing.assert_array_equal(observation[:, 1, 1], [1, 2, 3, 4, -1, 6])
        np.testing.assert_array_equal(observation[:, 1, 2], [7, 8, 9, 10, 11, 12])
        self.assertEqual(observation[Agent.TENSOR_LAYERS["WALL"], 0, 0], 1)
        self.assertEqual(observation[Agent.TENSOR_LAYERS["WALL"], 2, 2], 0)

    def test_get_coords_rejects_positions_with_wrong_length(self):
        agent = Agent(window_size=[3, 3], type_ids=self.TYPE_IDS)

        with self.assertRaisesRegex(ValueError, r"agent_pos must contain \[x, y\]"):
            agent._get_coords([1])

    def test_get_coords_rejects_positions_outside_memory_grid(self):
        agent = Agent(
            window_size=[3, 3],
            type_ids=self.TYPE_IDS,
            memory_grid_size=[4, 2],
        )

        for position in ([-1, 0], [4, 0], [0, -1], [0, 2]):
            with self.subTest(position=position):
                with self.assertRaisesRegex(
                    ValueError, "agent_pos is outside the memory grid"
                ):
                    agent._get_coords(position)

    def test_select_action_requires_a_loaded_model(self):
        agent = Agent(window_size=[3, 3], type_ids=self.TYPE_IDS)

        with self.assertRaisesRegex(RuntimeError, "No model is loaded"):
            agent.select_action([1, 1], ["up", "down"])

    def test_select_action_rejects_an_empty_action_list(self):
        agent = Agent(window_size=[3, 3], type_ids=self.TYPE_IDS)
        agent.model = Mock()

        with self.assertRaisesRegex(ValueError, "action_list must not be empty"):
            agent.select_action([1, 1], [])

        agent.model.predict.assert_not_called()

    def test_select_action_returns_action_selected_by_model(self):
        agent = Agent(window_size=[3, 3], type_ids=self.TYPE_IDS)
        agent.known_env_map[Agent.TENSOR_LAYERS["VISITED"], 1, 1] = 1
        agent.model = Mock()
        agent.model.predict.return_value = (np.array(1), None)

        action = agent.select_action([1, 1], ["left", "right"])

        self.assertEqual(action, "right")
        predicted_observation = agent.model.predict.call_args.args[0]
        np.testing.assert_array_equal(
            predicted_observation,
            agent.get_observation_tensor([1, 1]),
        )
        agent.model.predict.assert_called_once_with(
            predicted_observation,
            deterministic=True,
        )

    def test_select_action_runs_onnx_model_and_returns_highest_value_action(self):
        class FakeInferenceSession:
            def get_inputs(self):
                return [SimpleNamespace(name="observations")]

            def run(self, output_names, inputs):
                self.output_names = output_names
                self.inputs = inputs
                return [np.array([[0.1, 0.8, 0.2, 0.3]], dtype=np.float32)]

        agent = Agent(window_size=[3, 3], type_ids=self.TYPE_IDS)
        session = FakeInferenceSession()

        with patch("agent.agent.ort.InferenceSession", FakeInferenceSession):
            agent.model = session
            action = agent.select_action(
                [1, 1],
                ["up", "down", "left", "right"],
            )

        self.assertEqual(action, "down")
        self.assertIsNone(session.output_names)
        self.assertEqual(
            session.inputs["observations"].shape,
            (1, 6, 3, 3),
        )
        self.assertEqual(
            session.inputs["observations"].dtype,
            np.float32,
        )

    def test_select_action_rejects_model_index_outside_action_list(self):
        agent = Agent(window_size=[3, 3], type_ids=self.TYPE_IDS)
        agent.model = Mock()
        agent.model.predict.return_value = (np.array(2), None)

        with self.assertRaisesRegex(
            IndexError, "The model selected an action outside action_list"
        ):
            agent.select_action([1, 1], ["left", "right"])

    def test_update_memory_maps_observation_to_memory_layers(self):
        agent = Agent(
            window_size=[3, 3],
            type_ids=self.TYPE_IDS,
            memory_grid_size=[5, 5],
        )
        local_state = [
            [0, 1, 2],
            [3, 0, 0],
            [0, 0, 0],
        ]

        agent.update_memory([2, 2], local_state)

        known_map = agent.known_env_map
        np.testing.assert_array_equal(
            known_map[Agent.TENSOR_LAYERS["FINISH"], 1:4, 1:4],
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
        )
        self.assertEqual(known_map[Agent.TENSOR_LAYERS["VISITED"], 2, 2], 1)
        np.testing.assert_array_equal(
            known_map[Agent.TENSOR_LAYERS["WALL"], 1:4, 1:4],
            [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
        )
        np.testing.assert_array_equal(
            known_map[Agent.TENSOR_LAYERS["TRAVERSABLE"], 1:4, 1:4],
            [[1, 0, 1], [0, 1, 1], [1, 1, 1]],
        )
        np.testing.assert_array_equal(
            known_map[Agent.TENSOR_LAYERS["BONUS"], 1:4, 1:4],
            [[0, 0, 1], [0, 0, 0], [0, 0, 0]],
        )
        np.testing.assert_array_equal(
            known_map[Agent.TENSOR_LAYERS["REVEALED"], 1:4, 1:4],
            np.ones((3, 3), dtype=np.int32),
        )

    def test_update_memory_clips_observation_at_grid_edges(self):
        agent = Agent(
            window_size=[3, 3],
            type_ids=self.TYPE_IDS,
            memory_grid_size=[2, 2],
        )

        agent.update_memory([0, 0], [[1, 2, 3], [0, 1, 2], [3, 3, 1]])

        self.assertEqual(
            agent.known_env_map[Agent.TENSOR_LAYERS["FINISH"], 0, 0],
            1,
        )
        self.assertEqual(
            agent.known_env_map[Agent.TENSOR_LAYERS["BONUS"], 0, 1],
            1,
        )
        self.assertEqual(
            agent.known_env_map[Agent.TENSOR_LAYERS["WALL"], 1, 0],
            1,
        )
        self.assertEqual(
            agent.known_env_map[Agent.TENSOR_LAYERS["FINISH"], 1, 1],
            1,
        )
        self.assertEqual(
            agent.known_env_map[Agent.TENSOR_LAYERS["REVEALED"], 1, 1],
            1,
        )

    def test_update_memory_preserves_consumed_bonus(self):
        agent = Agent(
            window_size=[3, 3],
            type_ids=self.TYPE_IDS,
            memory_grid_size=[5, 5],
        )
        local_state = [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
        bonus_layer = Agent.TENSOR_LAYERS["BONUS"]

        agent.update_memory([2, 2], local_state)
        self.assertTrue(agent.consume_bonus([2, 2]))

        agent.update_memory([2, 2], local_state)

        self.assertEqual(agent.known_env_map[bonus_layer, 2, 2], -1)
        self.assertFalse(agent.consume_bonus([2, 2]))

    def test_update_memory_rejects_observation_with_wrong_dimensions(self):
        agent = Agent(window_size=[3, 3], type_ids=self.TYPE_IDS)

        with self.assertRaisesRegex(
            ValueError, "local_state dimensions must match window_size"
        ):
            agent.update_memory([1, 1], [[0, 0], [0, 0]])


if __name__ == "__main__":
    unittest.main()
