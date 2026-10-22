# Grid Env

`grid-env` is a small Python package for generating grid environments and
importing or exporting them as CSV. PNG rendering is available with the
optional `png` dependency.

## Install

```bash
python -m pip install .
```

To enable PNG output:

```bash
python -m pip install ".[png]"
```

## Usage

Install with the `png` extra to use `Png`; the CSV and stdout features require
no runtime dependencies.

```python
from grid_env import Exporter, GridEnv, Importer
from grid_env.output import Png, Stdout

environment = GridEnv()
environment.generate()

Stdout().call(environment)
Stdout().call(environment, path=[[0, 0], [1, 0]])
Exporter().call(environment, path_to_file="grid.csv")
Png().call(environment, path_to_file="grid.png")
Png().call(environment, path_to_file="grid-path.png", path=[[0, 0], [1, 0]])

loaded_grid = Importer().call(path_to_file="grid.csv")
```

## Testing
```
python -m unittest discover -s tests -v
```
