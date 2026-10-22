from typing import List, Optional

from grid_env.grid_env import GridEnv


class Stdout:
    def call(
        self,
        grid: GridEnv,
        path: Optional[List[List[int]]] = None,
    ) -> None:
        print("=== GridEnv Configuration ===")
        print(f"Size (Width, Height): {grid.width, grid.height}")
        print(f"Start Coordinate: {grid.start}")
        print(f"Actual Path Length: {grid.path_length}")
        print("TYPES:")
        for name, props in grid.TYPES.items():
            print(f"  - {name}: {props}")
        print(f"ACTIONS: {list(grid.ACTIONS.keys())}")
        print("\n=== Grid Matrix ===")
        path_cells = {tuple(coordinates) for coordinates in path or []}
        for y, row in enumerate(grid.grid):
            cells = []
            for x, cell in enumerate(row):
                formatted_cell = f"{cell:2d}"
                if (x, y) in path_cells:
                    formatted_cell = f"\033[48;2;144;238;144m{formatted_cell}\033[0m"
                cells.append(formatted_cell)
            print(" ".join(cells))
        print("============================")
