# TODO: extract class memory

from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
from stable_baselines3 import DQN

class Agent:
    REQUIRED_CELL_TYPES = {"empty", "finish", "bonus", "wall"}
    TENSOR_LAYERS = {
        "FINISH": 0,        # Finish Line (Detected finish coordinates)
        "VISITED": 1,       # Visited Memory (Historical trail of agent positions)
        "WALL": 2,          # Confirmed Walls (Detected obstacle locations)
        "TRAVERSABLE": 3,   # Confirmed Open Space (Verified safe floor/empty tiles)
        "BONUS": 4,         # Confirmed Bonus (Active, uncollected bonus tiles)
        "REVEALED": 5,      # Explored Space (1 = revealed/explored, 0 = unexplored / fog of war)
    }
    TENSOR_LAYERS_COUNT = len(TENSOR_LAYERS)

    def __init__(
        self,
        type_ids: Dict[str, int],
        window_size: List[int],
        memory_grid_size: Optional[List[int]] = [15, 15],
    ):
        self._validate_type_ids(type_ids)

        self.memory_grid_width, self.memory_grid_height = memory_grid_size
        self.window_size = window_size
        self.type_ids = type_ids
        self.model: Optional[DQN] = None
        self.known_env_map = self._init_env_map()


    @classmethod
    def _validate_type_ids(cls, type_ids: Dict[str, int]) -> None:
        missing_types = cls.REQUIRED_CELL_TYPES.difference(type_ids)
        if missing_types:
            missing = ", ".join(sorted(missing_types))
            raise ValueError("type_ids is missing required keys: {}".format(missing))


    def _init_env_map(self) -> np.ndarray:
        return np.zeros(
            (self.TENSOR_LAYERS_COUNT, self.memory_grid_height, self.memory_grid_width),
            dtype=np.int32,
        )


    def load_model(self, path: str) -> None:
        self.model = DQN.load(path)

    
    def reset(self) -> None:
        self.known_env_map = self._init_env_map()


    def get_observation_tensor(self, agent_pos: List[int]) -> np.ndarray:
        agent_x, agent_y = self._get_coords(agent_pos)
        window_width, window_height = self.window_size
        half_window_width = window_width // 2
        half_window_height = window_height // 2

        tensor = np.zeros(
            (self.TENSOR_LAYERS_COUNT, window_height, window_width),
            dtype=np.int32,
        )

        for channel_idx in range(self.TENSOR_LAYERS_COUNT):
            for dy_local, dy in enumerate(
                range(-half_window_height, half_window_height + 1)
            ):
                for dx_local, dx in enumerate(
                    range(-half_window_width, half_window_width + 1)
                ):
                    grid_x = agent_x + dx
                    grid_y = agent_y + dy

                    if (
                        0 <= grid_x < self.memory_grid_width
                        and 0 <= grid_y < self.memory_grid_height
                    ):
                        tensor[channel_idx, dy_local, dx_local] = self.known_env_map[
                            channel_idx, grid_y, grid_x
                        ]
                    elif channel_idx == self.TENSOR_LAYERS["WALL"]:
                        tensor[channel_idx, dy_local, dx_local] = 1
        return tensor


    def _get_coords(self, agent_pos: List[int]) -> Tuple[int, int]:
        if len(agent_pos) != 2:
            raise ValueError("agent_pos must contain [x, y].")
        agent_x, agent_y = agent_pos
        if not (
            0 <= agent_x < self.memory_grid_width
            and 0 <= agent_y < self.memory_grid_height
        ):
            raise ValueError("agent_pos is outside the memory grid.")
        return agent_x, agent_y


    def select_action(self, agent_pos: List[int], action_list: List[str]) -> str:
        if self.model is None:
            raise RuntimeError("No model is loaded.")
        if not action_list:
            raise ValueError("action_list must not be empty.")

        obs_tensor = self.get_observation_tensor(agent_pos)
        action_idx, _ = self.model.predict(obs_tensor, deterministic=True)
        index = int(np.asarray(action_idx).item())
        if not 0 <= index < len(action_list):
            raise IndexError("The model selected an action outside action_list.")
        return action_list[index]
    

    def update_memory(
        self,
        agent_pos: List[int],
        local_state: List[List[int]],
    ) -> None:
        """Update the global map with a centered local observation."""
        agent_x, agent_y = self._get_coords(agent_pos)
        window_width, window_height = self.window_size
        if len(local_state) != window_height or any(
            len(row) != window_width for row in local_state
        ):
            raise ValueError("local_state dimensions must match window_size.")

        half_window_width = window_width // 2
        half_window_height = window_height // 2

        finish_id = self.type_ids["finish"]
        wall_id = self.type_ids["wall"]
        bonus_id = self.type_ids["bonus"]

        self.known_env_map[self.TENSOR_LAYERS["VISITED"], agent_y, agent_x] = 1

        for dy_local, row in enumerate(local_state):
            for dx_local, cell_type in enumerate(row):
                grid_x = agent_x + (dx_local - half_window_width)
                grid_y = agent_y + (dy_local - half_window_height)

                if 0 <= grid_x < self.memory_grid_width and 0 <= grid_y < self.memory_grid_height:
                    self.known_env_map[self.TENSOR_LAYERS["FINISH"], grid_y, grid_x] = int(cell_type == finish_id)
                    self.known_env_map[self.TENSOR_LAYERS["WALL"], grid_y, grid_x] = int(cell_type == wall_id)
                    self.known_env_map[self.TENSOR_LAYERS["TRAVERSABLE"], grid_y, grid_x] = int(
                        cell_type in (self.type_ids["empty"], bonus_id)
                    )

                    if self.known_env_map[self.TENSOR_LAYERS["BONUS"], grid_y, grid_x] != -1:
                        self.known_env_map[self.TENSOR_LAYERS["BONUS"], grid_y, grid_x] = int(cell_type == bonus_id)

                    self.known_env_map[self.TENSOR_LAYERS["REVEALED"], grid_y, grid_x] = 1


    def consume_bonus(self, pos: List[int]) -> bool:
        grid_x, grid_y = self._get_coords(pos)
        if self.known_env_map[self.TENSOR_LAYERS["BONUS"], grid_y, grid_x] == 1:
            self.known_env_map[self.TENSOR_LAYERS["BONUS"], grid_y, grid_x] = -1
            return True
        return False
