import csv

from grid_env.grid_env import GridEnv


class Exporter:
    def call(
        self,
        grid: GridEnv,
        format: str = 'csv',
        path_to_file: str = 'grid.csv',
    ) -> None:
        if format.lower() != 'csv':
            raise ValueError(f"Unsupported export format: {format}")
        with open(path_to_file, mode='w', newline='') as f:
            f.write(f"# start: {grid.start[0]},{grid.start[1]}\n")
            f.write(f"# path_length: {grid.path_length}\n")
            writer = csv.writer(f)
            for row in grid.grid:
                writer.writerow(row)
