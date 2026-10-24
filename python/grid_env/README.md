# Grid Env

`grid-env` is a package for generating grid environments and
importing or exporting them as CSV. PNG rendering is available with the
optional `png` dependency.

CSV exports include `GridEnv.start` and `GridEnv.path_length` as comment
metadata before the grid rows. The importer restores these values and returns
a `GridEnv`.

## Usage

Install with the `png` extra to use `Png`; the CSV and stdout features require
no runtime dependencies.

```python
from grid_env import Exporter, GridEnv, Importer
from grid_env.output import Png, Stdout

environment = GridEnv()
environment.generate()

# display
Stdout().call(environment)
Stdout().call(environment, path=[[0, 0], [1, 0]])
Png().call(environment, path_to_file="grid.png")
Png().call(environment, path_to_file="grid-path.png", path=[[0, 0], [1, 0]])

# export/import
Exporter().call(environment, path_to_file="grid.csv")
loaded_environment = Importer().call(path_to_file="grid.csv")
```

## Testing
```
python -m unittest discover -s tests -v
```
