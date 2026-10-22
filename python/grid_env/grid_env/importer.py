import csv

from grid_env.grid_env import GridEnv


class Importer:
    def call(
        self,
        format: str = 'csv',
        path_to_file: str = 'grid.csv',
    ) -> GridEnv:
        if format.lower() != 'csv':
            raise ValueError(f"Unsupported import format: {format}")

        new_grid: list[list[int]] = []
        start = [0, 0]
        path_length = 0
        with open(path_to_file, mode='r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    if row[0].startswith("#"):
                        metadata, separator, value = row[0][1:].partition(":")
                        if not separator:
                            continue
                        metadata = metadata.strip()
                        if metadata == "start":
                            values = ",".join([value.strip(), *row[1:]])
                            try:
                                coordinates = [int(part) for part in values.split(",")]
                            except ValueError as error:
                                raise ValueError("Invalid start metadata.") from error
                            if len(coordinates) != 2:
                                raise ValueError("Start metadata must contain two coordinates.")
                            start = coordinates
                        elif metadata == "path_length":
                            try:
                                path_length = int(value.strip())
                            except ValueError as error:
                                raise ValueError("Invalid path_length metadata.") from error
                        continue
                    new_grid.append([int(val) for val in row])

        if not new_grid:
            raise ValueError("Imported file is empty or invalid.")

        grid = GridEnv([len(new_grid[0]), len(new_grid)])
        grid.grid = new_grid
        grid.start = start
        grid.path_length = path_length
        return grid