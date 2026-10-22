import random
from typing import List, Dict
from collections import deque


class GridEnv:
    TYPES = {
        'empty':  {'id': 0, 'count': None},
        'finish': {'id': 1, 'count': 1},
        'bonus':  {'id': 2, 'count': 5},
        'wall':   {'id': 3, 'count': 15},
    }

    ACTIONS: Dict[str, List[int]] = {
        'top': [0, -1],
        'bottom': [0, 1],
        'left': [-1, 0],
        'right': [1, 0]
    }

    def __init__(self, size: List[int] = [10, 10]):
        self.width, self.height = size
        self.grid = [
            [self.TYPES['empty']['id'] for _ in range(self.width)]
            for _ in range(self.height)
        ]
        self.start: List[int] = [0, 0]
        self.path: List[List[int]] = []
        self.path_length = 0

    def get_state(self, coords: List[int], window: List[int]) -> List[List[int]]:
        cx, cy = coords
        wx, wy = window
        hx = wx // 2
        hy = wy // 2

        sub_grid = []
        for dy in range(-hy, hy + 1):
            row = []
            for dx in range(-hx, hx + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    row.append(self.grid[ny][nx])
                else:
                    row.append(self.TYPES['wall']['id'])
            sub_grid.append(row)
        return sub_grid

    def generate(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Grid dimensions must be positive.")

        area = self.width * self.height
        path_length = (area + 1) // 2
        if path_length + self.TYPES['wall']['count'] > area:
            raise ValueError(
                "Grid is too small for the minimum path length and wall count."
            )
        if (
            self.TYPES['wall']['count']
            + self.TYPES['bonus']['count']
            + self.TYPES['finish']['count']
            > area
        ):
            raise ValueError("Grid is too small for the configured object counts.")

        self.grid = [
            [self.TYPES['empty']['id'] for _ in range(self.width)]
            for _ in range(self.height)
        ]

        self._generate_path(path_length)
        self._place_objects()

    def _generate_path(self, path_length: int) -> None:
        by_rows = random.choice([True, False])
        reverse_rows = random.choice([True, False])
        reverse_columns = random.choice([True, False])
        traversal: List[List[int]] = []
        if by_rows:
            ys = list(range(self.height))
            xs = list(range(self.width))
            if reverse_rows:
                ys.reverse()
            if reverse_columns:
                xs.reverse()
            for row_index, y in enumerate(ys):
                row_xs = xs if row_index % 2 == 0 else list(reversed(xs))
                traversal.extend([[x, y] for x in row_xs])
        else:
            xs = list(range(self.width))
            ys = list(range(self.height))
            if reverse_columns:
                xs.reverse()
            if reverse_rows:
                ys.reverse()
            for column_index, x in enumerate(xs):
                column_ys = ys if column_index % 2 == 0 else list(reversed(ys))
                traversal.extend([[x, y] for y in column_ys])

        rank = {(x, y): index for index, (x, y) in enumerate(traversal)}
        start = traversal[0]
        visited = {tuple(start)}
        path = [start]
        while len(path) < path_length:
            x, y = path[-1]
            neighbors = [
                [x + dx, y + dy]
                for dx, dy in self.ACTIONS.values()
                if 0 <= x + dx < self.width and 0 <= y + dy < self.height
                and (x + dx, y + dy) not in visited
            ]
            neighbors.sort(key=lambda coord: rank[tuple(coord)])
            if not neighbors:
                raise RuntimeError("DFS could not construct the required path.")
            next_cell = neighbors[0]
            path.append(next_cell)
            visited.add(tuple(next_cell))

        self.start = start
        self.path = path
        self.path_length = len(path)

    def _place_objects(self) -> None:
        self._place_walls()

        finish_x, finish_y = self.path[-1]
        self.grid[finish_y][finish_x] = self.TYPES['finish']['id']

        self._place_bonuses()

    def _place_walls(self) -> None:
        path_cells = {tuple(coords) for coords in self.path}
        wall_candidates = [
            [x, y]
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in path_cells
        ]
        for x, y in random.sample(wall_candidates, self.TYPES['wall']['count']):
            self.grid[y][x] = self.TYPES['wall']['id']

    def _place_bonuses(self) -> None:
        bonus_candidates = [
            [x, y]
            for y in range(self.height)
            for x in range(self.width)
            if self.grid[y][x] == self.TYPES['empty']['id']
        ]
        for x, y in random.sample(bonus_candidates, self.TYPES['bonus']['count']):
            self.grid[y][x] = self.TYPES['bonus']['id']
   