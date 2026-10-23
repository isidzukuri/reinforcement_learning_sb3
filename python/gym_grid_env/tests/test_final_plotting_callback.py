import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


_matplotlib = types.ModuleType("matplotlib")
_pyplot = types.ModuleType("matplotlib.pyplot")
_matplotlib.pyplot = _pyplot
try:
    importlib.import_module("matplotlib.pyplot")
except ModuleNotFoundError:
    for _function in ("subplots", "tight_layout", "show"):
        setattr(_pyplot, _function, Mock())
    sys.modules["matplotlib"] = _matplotlib
    sys.modules["matplotlib.pyplot"] = _pyplot

callback_module = importlib.import_module("gym_grid_env.final_plotting_callback")
FinalPlottingCallback = callback_module.FinalPlottingCallback


class FinalPlottingCallbackTests(unittest.TestCase):
    def test_initializes_with_environment_and_remaining_stage_count(self):
        env = SimpleNamespace(next_stages=[{}, {}])

        with patch("builtins.print"):
            callback = FinalPlottingCallback(env)

        self.assertIs(callback.env_ref, env)
        self.assertEqual(callback.last_seen_stages_left, 2)
        self.assertEqual(callback.update_interval, 100)
        self.assertEqual(callback.stage_transitions, [])

    def test_unwraps_vector_environment_to_first_environment(self):
        env = SimpleNamespace(next_stages=[{}])
        vector_env = SimpleNamespace(
            unwrapped=SimpleNamespace(envs=[env, SimpleNamespace()])
        )

        with patch("builtins.print"):
            callback = FinalPlottingCallback(vector_env)

        self.assertIs(callback.env_ref, env)
        self.assertEqual(callback.last_seen_stages_left, 1)

    def test_step_records_episode_means_and_stage_transition(self):
        env = SimpleNamespace(next_stages=[{}])
        with patch("builtins.print"):
            callback = FinalPlottingCallback(env, update_interval=10)

        callback.num_timesteps = 10
        callback.locals = {
            "infos": [
                {"episode": {"r": 20, "l": 8}},
                {"other": "info"},
                {"episode": {"r": 40, "l": 12}},
            ]
        }
        env.next_stages.clear()

        with patch("builtins.print"):
            should_continue = callback._on_step()

        self.assertTrue(should_continue)
        self.assertEqual(callback.stage_transitions, [(10, "Stages Left: 0")])
        self.assertEqual(callback.timesteps, [10])
        self.assertEqual(callback.ep_rewards, [30])
        self.assertEqual(callback.ep_lengths, [10])
        self.assertEqual(callback.rew_history, [20, 40])
        self.assertEqual(callback.len_history, [8, 12])

    def test_step_averages_only_the_most_recent_fifty_episodes(self):
        with patch("builtins.print"):
            callback = FinalPlottingCallback(
                SimpleNamespace(next_stages=[]),
                update_interval=1,
            )
        callback.num_timesteps = 1
        callback.locals = {
            "infos": [
                {"episode": {"r": reward, "l": reward}}
                for reward in range(1, 56)
            ]
        }

        callback._on_step()

        self.assertEqual(callback.ep_rewards, [30.5])
        self.assertEqual(callback.ep_lengths, [30.5])

    def test_training_end_plots_metrics_and_stage_transition_lines(self):
        env = SimpleNamespace(next_stages=[])
        with patch("builtins.print"):
            callback = FinalPlottingCallback(env)
        callback.timesteps = [100]
        callback.ep_rewards = [25]
        callback.ep_lengths = [10]
        callback.stage_transitions = [(50, "Stages Left: 1")]
        figure = Mock()
        reward_axis = Mock()
        length_axis = Mock()

        with (
            patch.object(
                callback_module.plt,
                "subplots",
                return_value=(figure, (reward_axis, length_axis)),
            ) as subplots,
            patch.object(callback_module.plt, "tight_layout") as layout,
            patch.object(callback_module.plt, "show") as show,
            patch("builtins.print"),
        ):
            callback._on_training_end()

        subplots.assert_called_once_with(2, 1, figsize=(12, 8), sharex=True)
        reward_axis.plot.assert_called_once_with(
            [100], [25], "b-", marker=".", label="Recent Reward Mean"
        )
        length_axis.plot.assert_called_once_with(
            [100], [10], "r-", marker=".", label="Mean Episode Length (Steps)"
        )
        for axis in (reward_axis, length_axis):
            axis.axvline(
                x=50,
                color="green",
                linestyle="--",
                linewidth=2,
                alpha=0.9,
                label="Stages Left: 1",
            )
        layout.assert_called_once_with()
        show.assert_called_once_with()

    def test_training_end_without_captured_history_does_not_plot(self):
        with patch("builtins.print") as print_output:
            callback = FinalPlottingCallback(SimpleNamespace(next_stages=[]))
            with patch.object(callback_module.plt, "subplots") as subplots:
                callback._on_training_end()

        subplots.assert_not_called()
        print_output.assert_any_call(
            "[FinalPlottingCallback] Error: No history captured. No plot to display."
        )


if __name__ == "__main__":
    unittest.main()
