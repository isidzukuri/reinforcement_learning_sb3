import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.model_manager import ModelManager


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "dqn_cnn_model.zip"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a Stable-Baselines3 DQN model to ONNX."
    )
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"saved model path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="ONNX output path (default: model path with a .onnx extension)",
    )
    args = parser.parse_args()

    output_path = args.output or args.model.with_suffix(".onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manager = ModelManager()
    model = manager.load_from_file(str(args.model))
    manager.export_to_onnx(model, str(output_path))
    print(f"Exported ONNX model to {output_path}")


if __name__ == "__main__":
    main()
