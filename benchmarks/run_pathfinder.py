from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
for package_dir in (
    PROJECT_ROOT / "python" / "pathfinder",
    PROJECT_ROOT / "python" / "agent",
    PROJECT_ROOT / "python" / "grid_env",
):
    sys.path.insert(0, str(package_dir))

from pathfinder import Pathfinder


def main() -> None:
    runtime_dir = Path(__file__).resolve().parent / "runtime"
    pathfinder = Pathfinder(
        str(runtime_dir / "grid_output.csv"),
        str(runtime_dir / "model.onnx"),
    )

    pathfinder.call()
    pathfinder.print_result()



if __name__ == "__main__":
    main()
