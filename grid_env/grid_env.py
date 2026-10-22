import random
from typing import List, Dict
from collections import deque

from grid_env.importer import Importer


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
        self.start: List[int] = [0, 0]
        self.path_length: int = 0
        empty_id = self.TYPES['empty']['id']
        self.grid = [[empty_id for _ in range(self.width)] for _ in range(self.height)]

    def place_objects(self):
        wall_id = self.TYPES['wall']['id']
        empty_id = self.TYPES['empty']['id']
        finish_id = self.TYPES['finish']['id']
        bonus_id = self.TYPES['bonus']['id']
        bonus_count = self.TYPES['bonus']['count']

        min_path_len = (self.width + self.height) // 2

        while True:
            # 1. Initialize grid with walls
            self.grid = [[wall_id for _ in range(self.width)] for _ in range(self.height)]

            if self.width < 3 or self.height < 3:
                for y in range(self.height):
                    for x in range(self.width):
                        self.grid[y][x] = empty_id
            else:
                # 2. DFS Maze Generation
                start_x, start_y = 1, 1
                self.grid[start_y][start_x] = empty_id
                stack = [(start_x, start_y)]

                while stack:
                    cx, cy = stack[-1]
                    neighbors = []
                    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                        nx, ny = cx + dx, cy + dy
                        if 0 < nx < self.width - 1 and 0 < ny < self.height - 1:
                            if self.grid[ny][nx] == wall_id:
                                neighbors.append((nx, ny))

                    if neighbors:
                        nx, ny = random.choice(neighbors)
                        mx, my = (cx + nx) // 2, (cy + ny) // 2
                        self.grid[my][mx] = empty_id
                        self.grid[ny][nx] = empty_id
                        stack.append((nx, ny))
                    else:
                        stack.pop()

            empty_cells = [[x, y] for y in range(self.height) for x in range(self.width) if self.grid[y][x] == empty_id]
            if len(empty_cells) < 2:
                continue

            # 3. Set start coordinate
            self.start = random.choice(empty_cells)

            # 4. Check minimum path length condition and find finish
            distances = self._bfs_distances(self.start)
            valid_finishes = [coord for coord, dist in distances.items() if dist >= min_path_len]

            if not valid_finishes:
                if distances:
                    farthest_coord = max(distances, key=distances.get)
                    if distances[farthest_coord] >= min_path_len // 2:
                        finish_coord = farthest_coord
                        self.path_length = distances[finish_coord]
                    else:
                        continue
                else:
                    continue
            else:
                finish_coord = random.choice(valid_finishes)
                self.path_length = distances[finish_coord]

            self.grid[finish_coord[1]][finish_coord[0]] = finish_id

            # 5. Place bonus items
            remaining_empty = [[x, y] for y in range(self.height) for x in range(self.width) if self.grid[y][x] == empty_id and [x, y] != self.start]
            if bonus_count is not None and len(remaining_empty) >= bonus_count:
                bonus_coords = random.sample(remaining_empty, bonus_count)
                for bx, by in bonus_coords:
                    self.grid[by][bx] = bonus_id

            break

    def get_state(self, coords: List[int], window: List[int]) -> List[List[int]]:
        cx, cy = coords
        wx, wy = window
        hx = wx // 2
        hy = wy // 2

        wall_id = self.TYPES['wall']['id']
        sub_grid = []
        for dy in range(-hy, hy + 1):
            row = []
            for dx in range(-hx, hx + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    row.append(self.grid[ny][nx])
                else:
                    row.append(wall_id)
            sub_grid.append(row)
        return sub_grid


    def _bfs_distances(self, start_coord: List[int]) -> Dict[tuple, int]:
            wall_id = self.TYPES['wall']['id']
            queue = deque([(start_coord[0], start_coord[1], 0)])
            visited = {(start_coord[0], start_coord[1])}
            distances = {(start_coord[0], start_coord[1]): 0}
    
            while queue:
                cx, cy, dist = queue.popleft()
                for action in self.ACTIONS.values():
                    nx, ny = cx + action[0], cy + action[1]
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        if self.grid[ny][nx] != wall_id and (nx, ny) not in visited:
                            visited.add((nx, ny))
                            distances[(nx, ny)] = dist + 1
                            queue.append((nx, ny, dist + 1))
            return distances
    
