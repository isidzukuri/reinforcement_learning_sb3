import csv


class Importer:
    def call(
        self,
        format: str = 'csv',
        path_to_file: str = 'grid.csv',
    ) -> list[list[int]]:
        if format.lower() != 'csv':
            raise ValueError(f"Unsupported import format: {format}")

        new_grid = []
        with open(path_to_file, mode='r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    new_grid.append([int(val) for val in row])

        if not new_grid:
            raise ValueError("Imported file is empty or invalid.")

        return new_grid