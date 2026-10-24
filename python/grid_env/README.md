# Grid Env

`grid-env` is a package for generating grid environments and importing or exporting them as CSV.


## Usage

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
