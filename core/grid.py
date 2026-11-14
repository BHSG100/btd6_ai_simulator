from __future__ import annotations

from enum import IntEnum
from typing import Tuple
import numpy as np
import numpy.typing as npt


class CellState(IntEnum):
    EMPTY = 0
    PATH = 1
    TOWER = 2
    OBSTACLE = 3


class GridMap:
    """Representação espacial 2D baseada em NumPy (np.int8)."""

    def __init__(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("As dimensões do grid devem ser maiores que zero.")

        self.width: int = width
        self.height: int = height
        self.grid: npt.NDArray[np.int8] = np.full(
            (width, height), CellState.EMPTY, dtype=np.int8
        )

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_cell_free(self, x: int, y: int) -> bool:
        if not self.in_bounds(x, y):
            return False
        return bool(self.grid[x, y] == CellState.EMPTY)

    def is_path(self, x: int, y: int) -> bool:
        if not self.in_bounds(x, y):
            return False
        return bool(self.grid[x, y] == CellState.PATH)

    def set_path_cells(self, path_coords: npt.NDArray[np.int32]) -> None:
        if path_coords.size > 0:
            self.grid[path_coords[:, 0], path_coords[:, 1]] = CellState.PATH

    def place_tower_footprint(self, x: int, y: int) -> bool:
        if not self.is_cell_free(x, y):
            return False
        self.grid[x, y] = CellState.TOWER
        return True

    def clear_tower_footprint(self, x: int, y: int) -> None:
        if self.in_bounds(x, y) and self.grid[x, y] == CellState.TOWER:
            self.grid[x, y] = CellState.EMPTY

    @staticmethod
    def euclidean_distance(
        tower_pos: Tuple[float, float] | npt.NDArray[np.float32],
        target_pos: Tuple[float, float] | npt.NDArray[np.float32],
    ) -> float:
        dx = float(tower_pos[0]) - float(target_pos[0])
        dy = float(tower_pos[1]) - float(target_pos[1])
        return float(np.hypot(dx, dy))

    @staticmethod
    def vectorized_distances(
        tower_pos: npt.NDArray[np.float32],
        path_points: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        deltas = path_points - tower_pos
        return np.sqrt(np.einsum("ij,ij->i", deltas, deltas, dtype=np.float32))

    def get_cells_in_range_mask(
        self, center_x: float, center_y: float, radius: float
    ) -> npt.NDArray[np.bool_]:
        x_indices, y_indices = np.ogrid[: self.width, : self.height]
        dist_sq = (x_indices - center_x) ** 2 + (y_indices - center_y) ** 2
        return dist_sq <= (radius ** 2)
