from typing import List

from agent import Agent, ModelManager
from grid_env import GridEnv, Importer
from grid_env.output.stdout import Stdout


class Pathfinder:
    AGENT_WINDOW_SIZE = [5, 5]
    ACTIONS_LIMIT = 2000

    def __init__(
        self,
        grid_file_path: str,
        model_file_path: str,
    ) -> None:
        self.grid_file_path = grid_file_path
        self.model_file_path = model_file_path
        self.lock = False
        self.grid_env = None
        self.agent = None
        self.timesteps = 0
        self.path = []

    def call(self) -> bool:
        self._prepare_runtime()

        action_list = list(GridEnv.ACTIONS.keys())
        finish_id = GridEnv.TYPES["finish"]["id"]
        agent_pos = self.grid_env.start.copy()
        self.path.append(agent_pos.copy())

        for _ in range(self.ACTIONS_LIMIT):
            if self.grid_env.grid[agent_pos[1]][agent_pos[0]] == finish_id:
                return True

            self.timesteps += 1

            grid_chunk = self.grid_env.get_state(
                agent_pos,
                self.AGENT_WINDOW_SIZE,
            )
            self.agent.update_memory(agent_pos, grid_chunk)
            action = self.agent.select_action(agent_pos, action_list)

            dx, dy = GridEnv.ACTIONS[action]
            next_pos = [agent_pos[0] + dx, agent_pos[1] + dy]
            if self._is_valid_position(next_pos):
                agent_pos = next_pos
                self.path.append(agent_pos.copy())

        self._raise_time_limit_error()

    def _prepare_runtime(self):
        if self.lock:
            self.agent.reset()
        else:
            self._load_grid()
            self._construct_agent()

        self.lock = True
        self.timesteps = 0
        self.path = []

    def _load_grid(self):
        self.grid_env = Importer().call(path_to_file=self.grid_file_path)

    def _construct_agent(self):
        type_ids = {
            name: cell_type["id"]
            for name, cell_type in GridEnv.TYPES.items()
        }

        self.agent = Agent(
            type_ids=type_ids,
            window_size=self.AGENT_WINDOW_SIZE,
            memory_grid_size=[self.grid_env.width, self.grid_env.height],
        )
        self.agent.model = ModelManager().load_onnx(self.model_file_path)

    def _is_valid_position(self, position) -> bool:
        return (
            0 <= position[0] < self.grid_env.width
            and 0 <= position[1] < self.grid_env.height
            and self.grid_env.grid[position[1]][position[0]]
            != GridEnv.TYPES["wall"]["id"]
        )

    def _raise_time_limit_error(self):
        raise RuntimeError(
            f"Agent did not reach the finish within {self.ACTIONS_LIMIT} actions."
        )

    def result(self):
        return {
            "timesteps": self.timesteps,
            "path": self.path,
        }
    
    def print_result(self):
        print(f"Path is found after {self.timesteps} iterations")

        Stdout().call(self.grid_env, path=self.path)
