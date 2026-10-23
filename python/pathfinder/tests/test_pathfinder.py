import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pathfinder import Pathfinder


class PathfinderTests(unittest.TestCase):
    @patch("pathfinder.pathfinder.ModelManager")
    @patch("pathfinder.pathfinder.Agent")
    @patch("pathfinder.pathfinder.Importer")
    def test_call_moves_agent_until_it_reaches_finish(
        self,
        importer_class,
        agent_class,
        model_manager_class,
    ):
        grid = SimpleNamespace(
            width=3,
            height=1,
            start=[0, 0],
            grid=[[0, 0, 1]],
            get_state=Mock(return_value=[[0] * 5 for _ in range(5)]),
        )
        importer_class.return_value.call.return_value = grid
        agent = agent_class.return_value
        agent.select_action.side_effect = ["right", "right"]
        model = object()
        model_manager_class.return_value.load_onnx.return_value = model

        pathfinder = Pathfinder("level.csv", "model.onnx")
        pathfinder.call()

        importer_class.return_value.call.assert_called_once_with(
            path_to_file="level.csv"
        )
        agent_class.assert_called_once_with(
            type_ids={"empty": 0, "finish": 1, "bonus": 2, "wall": 3},
            window_size=[5, 5],
            memory_grid_size=[3, 1],
        )
        model_manager_class.return_value.load_onnx.assert_called_once_with(
            "model.onnx"
        )
        self.assertIs(agent.model, model)
        self.assertEqual(pathfinder.path, [[0, 0], [1, 0], [2, 0]])
        self.assertEqual(agent.select_action.call_count, 2)
        self.assertEqual(agent.update_memory.call_count, 2)

    @patch("pathfinder.pathfinder.ModelManager")
    @patch("pathfinder.pathfinder.Agent")
    @patch("pathfinder.pathfinder.Importer")
    def test_call_stops_with_error_when_action_limit_is_reached(
        self,
        importer_class,
        agent_class,
        model_manager_class,
    ):
        grid = SimpleNamespace(
            width=2,
            height=1,
            start=[0, 0],
            grid=[[0, 3]],
            get_state=Mock(return_value=[[0] * 5 for _ in range(5)]),
        )
        importer_class.return_value.call.return_value = grid
        agent_class.return_value.select_action.return_value = "right"
        model_manager_class.return_value.load_onnx.return_value = object()
        pathfinder = Pathfinder("level.csv", "model.onnx")
        pathfinder.ACTIONS_LIMIT = 2

        with self.assertRaisesRegex(
            RuntimeError,
            "Agent did not reach the finish within 2 actions",
        ):
            pathfinder.call()

        self.assertEqual(pathfinder.path, [[0, 0]])


if __name__ == "__main__":
    unittest.main()
