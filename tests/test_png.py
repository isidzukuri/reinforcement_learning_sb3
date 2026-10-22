import sys
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock, call, patch

from grid_env import GridEnv
from grid_env.output import Png


class PngTests(unittest.TestCase):
    def setUp(self):
        self.grid = GridEnv([5, 1])
        self.grid.grid = [[0, 1, 2, 3, 99]]
        self.grid.start = [2, 0]
        self.grid.path_length = 4

    def test_renders_cells_and_saves_to_default_path(self):
        image = Mock()
        drawer = Mock()
        image_module = types.ModuleType("PIL.Image")
        image_module.new = Mock(return_value=image)
        image_draw_module = types.ModuleType("PIL.ImageDraw")
        image_draw_module.Draw = Mock(return_value=drawer)
        pil_module = types.ModuleType("PIL")
        pil_module.Image = image_module
        pil_module.ImageDraw = image_draw_module
        output = StringIO()

        with patch.dict(
            sys.modules,
            {
                "PIL": pil_module,
                "PIL.Image": image_module,
                "PIL.ImageDraw": image_draw_module,
            },
        ), redirect_stdout(output):
            Png().call(self.grid)

        image_module.new.assert_called_once_with("RGB", (200, 40), "white")
        image_draw_module.Draw.assert_called_once_with(image)
        self.assertEqual(
            drawer.rectangle.call_args_list,
            [
                call([0, 0, 40, 40], fill=(240, 240, 240), outline=(200, 200, 200)),
                call([40, 0, 80, 40], fill=(46, 204, 113), outline=(200, 200, 200)),
                call([80, 0, 120, 40], fill=(241, 196, 15), outline=(200, 200, 200)),
                call([120, 0, 160, 40], fill=(52, 73, 94), outline=(200, 200, 200)),
                call([160, 0, 200, 40], fill=(255, 255, 255), outline=(200, 200, 200)),
            ],
        )
        self.assertEqual(
            drawer.text.call_args_list,
            [
                call((15, 12), "0", fill=(0, 0, 0)),
                call((55, 12), "1", fill=(0, 0, 0)),
                call((95, 12), "2", fill=(0, 0, 0)),
                call((135, 12), "3", fill=(0, 0, 0)),
                call((175, 12), "99", fill=(0, 0, 0)),
            ],
        )
        image.save.assert_called_once_with("grid_output.png")
        self.assertEqual(
            output.getvalue(),
            "Grid image successfully saved to grid_output.png\n",
        )

    def test_saves_to_supplied_path(self):
        image = Mock()
        image_module = types.ModuleType("PIL.Image")
        image_module.new = Mock(return_value=image)
        image_draw_module = types.ModuleType("PIL.ImageDraw")
        image_draw_module.Draw = Mock(return_value=Mock())
        pil_module = types.ModuleType("PIL")
        pil_module.Image = image_module
        pil_module.ImageDraw = image_draw_module

        with patch.dict(
            sys.modules,
            {
                "PIL": pil_module,
                "PIL.Image": image_module,
                "PIL.ImageDraw": image_draw_module,
            },
        ), patch("builtins.print") as print_output:
            Png().call(self.grid, "custom.png")

        image.save.assert_called_once_with("custom.png")
        print_output.assert_called_once_with(
            "Grid image successfully saved to custom.png"
        )

    def test_highlights_path_coordinates(self):
        image = Mock()
        drawer = Mock()
        image_module = types.ModuleType("PIL.Image")
        image_module.new = Mock(return_value=image)
        image_draw_module = types.ModuleType("PIL.ImageDraw")
        image_draw_module.Draw = Mock(return_value=drawer)
        pil_module = types.ModuleType("PIL")
        pil_module.Image = image_module
        pil_module.ImageDraw = image_draw_module

        with patch.dict(
            sys.modules,
            {
                "PIL": pil_module,
                "PIL.Image": image_module,
                "PIL.ImageDraw": image_draw_module,
            },
        ), patch("builtins.print"):
            Png().call(self.grid, path=[[2, 0]])

        self.assertEqual(
            drawer.rectangle.call_args_list[2],
            call([80, 0, 120, 40], fill=(144, 238, 144), outline=(200, 200, 200)),
        )

    def test_reports_missing_pillow_dependency(self):
        original_import = __import__

        def import_without_pillow(name, *args, **kwargs):
            if name == "PIL":
                raise ImportError("Pillow is not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_pillow):
            with self.assertRaisesRegex(
                ImportError,
                "Pillow library is required for PNG output.*pip install pillow",
            ):
                Png().call(self.grid)


if __name__ == "__main__":
    unittest.main()
