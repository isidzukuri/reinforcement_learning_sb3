from grid_env.grid_env import GridEnv


class Stdout:
    def call(self, grid: GridEnv) -> None:
        print("=== GridEnv Configuration ===")
        print(f"Size (Width, Height): {grid.width, grid.height}")
        print(f"Start Coordinate: {grid.start}")
        print(f"Actual Path Length: {grid.path_length}")
        print("TYPES:")
        for name, props in grid.TYPES.items():
            print(f"  - {name}: {props}")
        print(f"ACTIONS: {list(grid.ACTIONS.keys())}")
        print("\n=== Grid Matrix ===")
        for row in grid.grid:
            print(" ".join(f"{cell:2d}" for cell in row))
        print("============================")
