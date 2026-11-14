from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from enum import IntEnum
from typing import Dict, List, Sequence
import numpy as np
import numpy.typing as npt

from core.arenas import TargetingMode
from core.engine import GameEngine
from utils.data_loader import BloonLookupTable, CompiledRound, TowerData


class GeneOp(IntEnum):
    PLACE = 0    # [round_idx, 0, tower_type_idx, x, y, target_mode]
    UPGRADE = 1  # [round_idx, 1, tower_slot, path_idx, target_mode, 0]
    SELL = 2     # [round_idx, 2, tower_slot, 0, 0, 0]


_WORKER_ENGINE: GameEngine | None = None
_WORKER_TOWER_IDS: List[str] = []


def _init_worker(
    grid_width: int,
    grid_height: int,
    waypoints: npt.NDArray[np.float32],
    path_cells: npt.NDArray[np.int32],
    towers_catalog: Dict[str, TowerData],
    bloon_lookup: BloonLookupTable,
    rounds_schedule: List[CompiledRound],
) -> None:
    global _WORKER_ENGINE, _WORKER_TOWER_IDS
    _WORKER_TOWER_IDS = sorted(towers_catalog.keys())
    _WORKER_ENGINE = GameEngine(
        grid_width=grid_width,
        grid_height=grid_height,
        waypoints=waypoints,
        path_cells=path_cells,
        towers_catalog=towers_catalog,
        bloon_lookup=bloon_lookup,
        rounds_schedule=rounds_schedule,
    )


def _evaluate_single_genome(
    genome: npt.NDArray[np.int32], dt: float = 0.05
) -> float:
    """
    Executa um episódio completo guiado pelo cromossomo polimórfico na engine local.
    Formato do Genoma: shape (G, 6) -> [round_idx, op_type, arg1, arg2, arg3, arg4]
      - GeneOp.PLACE (0):   [round, 0, tower_type_idx, x, y, target_mode]
      - GeneOp.UPGRADE (1): [round, 1, tower_slot, path_idx (0..2), target_mode (-1=manter), 0]
      - GeneOp.SELL (2):    [round, 2, tower_slot, 0, 0, 0]
    """
    if _WORKER_ENGINE is None:
        raise RuntimeError("Worker não inicializado corretamente.")

    engine = _WORKER_ENGINE
    engine.reset()
    num_tower_types = len(_WORKER_TOWER_IDS)

    if genome.ndim == 2 and genome.shape[0] > 0:
        order = np.argsort(genome[:, 0], kind="mergesort")
        sorted_genes = genome[order]
    else:
        sorted_genes = np.empty((0, 6), dtype=np.int32)

    gene_cursor = 0
    num_genes = sorted_genes.shape[0]

    while not engine.is_game_over and not engine.is_victory:
        current_r = engine.current_round_idx

        while gene_cursor < num_genes and sorted_genes[gene_cursor, 0] <= current_r:
            gene = sorted_genes[gene_cursor]
            op = int(gene[1])

            if op == GeneOp.PLACE:
                t_idx = int(gene[2])
                if 0 <= t_idx < num_tower_types:
                    t_id = _WORKER_TOWER_IDS[t_idx]
                    mode_val = int(gene[5]) if gene.shape[0] > 5 else 0
                    safe_mode = TargetingMode(int(np.clip(mode_val, 0, 3)))
                    engine.action_place_tower(
                        tower_id=t_id,
                        x=int(gene[3]),
                        y=int(gene[4]),
                        target_mode=safe_mode,
                    )

            elif op == GeneOp.UPGRADE:
                slot = int(gene[2])
                path_idx = int(gene[3])
                if 0 <= path_idx <= 2:
                    engine.action_upgrade_tower(tower_slot=slot, path_idx=path_idx)
                    if gene.shape[0] > 4 and int(gene[4]) >= 0:
                        safe_mode = TargetingMode(int(np.clip(gene[4], 0, 3)))
                        engine.action_change_targeting(slot, safe_mode)

            elif op == GeneOp.SELL:
                slot = int(gene[2])
                engine.action_sell_tower(tower_slot=slot)

            gene_cursor += 1

        engine.run_round(dt=dt)

    fitness = (
        (engine.current_round_idx * 10000.0)
        + (engine.total_pops * 1.0)
        + (engine.lives * 15.0)
    )
    if engine.is_victory:
        fitness += 100000.0 + (engine.cash * 2.0)

    return float(fitness)


class ParallelGAEvaluator:
    """Gerenciador de avaliação paralela para populações de Algoritmos Genéticos."""

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        waypoints: npt.NDArray[np.float32],
        path_cells: npt.NDArray[np.int32],
        towers_catalog: Dict[str, TowerData],
        bloon_lookup: BloonLookupTable,
        rounds_schedule: List[CompiledRound],
        max_workers: int | None = None,
        dt: float = 0.05,
    ) -> None:
        self.dt = dt
        self.executor = ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(
                grid_width,
                grid_height,
                waypoints,
                path_cells,
                towers_catalog,
                bloon_lookup,
                rounds_schedule,
            ),
        )

    def evaluate_population(
        self, population: Sequence[npt.NDArray[np.int32]], chunksize: int = 8
    ) -> npt.NDArray[np.float32]:
        futures = list(
            self.executor.map(
                _evaluate_single_genome,
                population,
                [self.dt] * len(population),
                chunksize=chunksize,
            )
        )
        return np.array(futures, dtype=np.float32)

    def close(self) -> None:
        self.executor.shutdown(wait=True)

    def __enter__(self) -> ParallelGAEvaluator:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
