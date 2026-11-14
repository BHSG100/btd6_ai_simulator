from __future__ import annotations

import numpy as np
import numpy.typing as npt


class WaypointPath:
    """Representa a trilha do mapa parametrizada pela distância acumulada 1D."""

    def __init__(self, waypoints: npt.NDArray[np.float32]) -> None:
        if waypoints.ndim != 2 or waypoints.shape[1] != 2 or waypoints.shape[0] < 2:
            raise ValueError("waypoints deve ter shape (N, 2) com pelo menos 2 pontos.")

        self.waypoints: npt.NDArray[np.float32] = waypoints.astype(np.float32)

        segment_vectors = np.diff(self.waypoints, axis=0)
        self.segment_lengths: npt.NDArray[np.float32] = np.sqrt(
            np.einsum("ij,ij->i", segment_vectors, segment_vectors)
        )

        self.cumulative_distances: npt.NDArray[np.float32] = np.concatenate(
            ([np.float32(0.0)], np.cumsum(self.segment_lengths, dtype=np.float32))
        )
        self.total_length: float = float(self.cumulative_distances[-1])

        self.unit_directions: npt.NDArray[np.float32] = (
            segment_vectors / self.segment_lengths[:, np.newaxis]
        )

    def distances_to_coords(
        self, distances: npt.NDArray[np.float32], out: npt.NDArray[np.float32] | None = None
    ) -> npt.NDArray[np.float32]:
        seg_indices = np.searchsorted(self.cumulative_distances, distances, side="right") - 1
        max_seg_idx = self.waypoints.shape[0] - 2
        np.clip(seg_indices, 0, max_seg_idx, out=seg_indices)

        local_dist = distances - self.cumulative_distances[seg_indices]

        if out is None:
            out = np.empty((distances.shape[0], 2), dtype=np.float32)

        np.multiply(
            self.unit_directions[seg_indices],
            local_dist[:, np.newaxis],
            out=out,
        )
        np.add(self.waypoints[seg_indices], out, out=out)
        return out
