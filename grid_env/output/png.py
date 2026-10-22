from typing import Optional

from grid_env.grid_env import GridEnv


class Png:
    def call(
        self,
        grid: GridEnv,
        path_to_file: Optional[str] = None,
    ) -> None:
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            raise ImportError(
                "Pillow library is required for PNG output. "
                "Install it using 'pip install pillow'."
            )

        output_path = path_to_file if path_to_file is not None else 'grid_output.png'
        cell_size = 40
        image = Image.new(
            "RGB",
            (grid.width * cell_size, grid.height * cell_size),
            "white",
        )
        draw = ImageDraw.Draw(image)
        colors = {
            grid.TYPES['empty']['id']: (240, 240, 240),
            grid.TYPES['finish']['id']: (46, 204, 113),
            grid.TYPES['bonus']['id']: (241, 196, 15),
            grid.TYPES['wall']['id']: (52, 73, 94),
        }

        for y in range(grid.height):
            for x in range(grid.width):
                cell_type = grid.grid[y][x]
                color = colors.get(cell_type, (255, 255, 255))
                box = [
                    x * cell_size,
                    y * cell_size,
                    (x + 1) * cell_size,
                    (y + 1) * cell_size,
                ]
                draw.rectangle(box, fill=color, outline=(200, 200, 200))
                draw.text(
                    (x * cell_size + 15, y * cell_size + 12),
                    str(cell_type),
                    fill=(0, 0, 0),
                )

        image.save(output_path)
        print(f"Grid image successfully saved to {output_path}")
