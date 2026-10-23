from pathlib import Path
import sys

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "python" / "grid_env"),
)

from grid_env import Exporter, GridEnv
from grid_env.output import Png, Stdout


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "runtime"
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = GridEnv([15, 15])
    grid.generate()

    Stdout().call(grid)
    Png().call(grid, path_to_file=str(output_dir / "grid_output.png"))
    Exporter().call(grid, path_to_file=str(output_dir / "grid_output.csv"))


if __name__ == "__main__":
    main()
