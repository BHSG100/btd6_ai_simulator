from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import numpy.typing as npt

from core.arenas import BloonPool, TargetingMode, TowerManager
from core.grid import CellState, GridMap
from core.jit_kernels import (
    jit_run_round_loop,
    jit_spawn_step,
    jit_step_combat_v2,
    jit_step_movement,
)
from core.path import WaypointPath
from utils.data_loader import BloonLookupTable, CompiledRound, TowerData


@dataclass(frozen=True)
class StepResult:
    """Métricas de retorno imediato para cálculo de recompensa (Reward Shaping)."""
    cash_earned: int
    pops: int
    lives_lost: int
    round_cleared: bool
    is_game_over: bool
    is_victory: bool


class GameEngine:
    """
    Fachada principal da simulação BTD6 para treinamento massivo em IA.
    Suporta compra, upgrades dinâmicos com trava 5-2-0, venda de torres
    e execução compilada LLVM via Numba (@njit).
    """

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        waypoints: npt.NDArray[np.float32],
        path_cells: npt.NDArray[np.int32],
        towers_catalog: Dict[str, TowerData],
        bloon_lookup: BloonLookupTable,
        rounds_schedule: List[CompiledRound],
        starting_cash: int = 650,
        starting_lives: int = 100,
        max_bloons: int = 4096,
        max_towers: int = 64,
        sell_refund_rate: float = 0.70,
    ) -> None:
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.waypoints = waypoints
        self.path_cells = path_cells
        self.towers_catalog = towers_catalog
        self.bloon_lookup = bloon_lookup
        self.rounds_schedule = rounds_schedule

        self.starting_cash = starting_cash
        self.starting_lives = starting_lives
        self.max_bloons = max_bloons
        self.max_towers = max_towers
        self.sell_refund_rate = sell_refund_rate

        self._scratch_cand = np.empty(max_bloons, dtype=np.int32)
        self._scratch_hits = np.empty(max_bloons, dtype=np.int32)
        self._scratch_queue = np.empty(max_bloons, dtype=np.int32)

        self.grid: GridMap
        self.path: WaypointPath
        self.bloons: BloonPool
        self.towers: TowerManager

        self.cash: int = starting_cash
        self.lives: int = starting_lives
        self.current_round_idx: int = 0
        self.total_pops: int = 0
        self.ticks_elapsed: int = 0
        self.round_active: bool = False
        self.round_elapsed_time: float = 0.0
        self.round_spawn_cursor: int = 0
        self.is_game_over: bool = False
        self.is_victory: bool = False

        self.reset()

    def reset(self) -> npt.NDArray[np.float32]:
        """Reinicia o estado da engine para um novo episódio de treinamento."""
        self.grid = GridMap(self.grid_width, self.grid_height)
        self.grid.set_path_cells(self.path_cells)

        self.path = WaypointPath(self.waypoints)
        self.bloons = BloonPool(self.max_bloons)
        self.towers = TowerManager(self.max_towers, self.sell_refund_rate)

        self.cash = self.starting_cash
        self.lives = self.starting_lives
        self.current_round_idx = 0
        self.total_pops = 0
        self.ticks_elapsed = 0
        self.round_active = False
        self.round_elapsed_time = 0.0
        self.round_spawn_cursor = 0
        self.is_game_over = False
        self.is_victory = False

        return self.get_observation_vector()

    # =========================================================================
    # API DE AÇÕES DA IA (Posicionamento, Upgrades 5-2-0, Venda e Mira)
    # =========================================================================

    def action_place_tower(
        self,
        tower_id: str,
        x: int,
        y: int,
        top_tier: int = 0,
        mid_tier: int = 0,
        bot_tier: int = 0,
        target_mode: TargetingMode = TargetingMode.FIRST,
    ) -> int:
        if self.is_game_over or tower_id not in self.towers_catalog:
            return -1

        t_data = self.towers_catalog[tower_id]
        try:
            stats = t_data.compute_full_stats(top_tier, mid_tier, bot_tier)
        except ValueError:
            return -1

        if self.cash < stats.total_cost:
            return -1

        if not self.grid.place_tower_footprint(x, y):
            return -1

        slot = self.towers.add_configured_tower(
            tower_id=tower_id,
            x=x,
            y=y,
            stats=stats,
            tiers=(top_tier, mid_tier, bot_tier),
            target_mode=target_mode,
        )
        if slot == -1:
            self.grid.grid[x, y] = CellState.EMPTY
            return -1

        self.cash -= stats.total_cost
        return slot

    def action_upgrade_tower(self, tower_slot: int, path_idx: int) -> bool:
        """
        Evolui o próximo tier do caminho `path_idx` (0=Top, 1=Mid, 2=Bot)
        na torre do `tower_slot`, respeitando crosspathing 5-2-0 e caixa.
        """
        if self.is_game_over or not (0 <= tower_slot < self.max_towers):
            return False

        t_id = self.towers.tower_ids[tower_slot]
        if t_id is None or not self.towers.arena.active[tower_slot]:
            return False

        cost = self.towers.upgrade_tower(
            slot=tower_slot,
            path_idx=path_idx,
            tower_data=self.towers_catalog[t_id],
            available_cash=self.cash,
        )
        if cost < 0:
            return False

        self.cash -= cost
        return True

    def action_sell_tower(self, tower_slot: int) -> int:
        """
        Vende a torre no `tower_slot`, reembolsa 70% do investimento total
        e libera a célula correspondente no GridMap.
        """
        if self.is_game_over or not (0 <= tower_slot < self.max_towers):
            return -1

        refund, gx, gy = self.towers.sell_tower(tower_slot)
        if refund < 0:
            return -1

        self.grid.clear_tower_footprint(gx, gy)
        self.cash += refund
        return refund

    def action_change_targeting(
        self, tower_slot: int, new_mode: TargetingMode
    ) -> bool:
        if 0 <= tower_slot < self.max_towers and self.towers.arena.active[tower_slot]:
            self.towers.arena.target_mode[tower_slot] = int(new_mode)
            return True
        return False

    # =========================================================================
    # EXECUÇÃO DA SIMULAÇÃO
    # =========================================================================

    def start_next_round(self) -> bool:
        if self.round_active or self.is_game_over or self.is_victory:
            return False
        if self.current_round_idx >= len(self.rounds_schedule):
            self.is_victory = True
            return False

        self.round_elapsed_time = 0.0
        self.round_spawn_cursor = 0
        self.round_active = True
        return True

    def step_tick(self, dt: float = 0.05) -> StepResult:
        if self.is_game_over or self.is_victory or not self.round_active:
            return StepResult(0, 0, 0, False, self.is_game_over, self.is_victory)

        c_round = self.rounds_schedule[self.current_round_idx]
        self.ticks_elapsed += 1

        lives_lost, _ = jit_step_movement(
            float(dt),
            self.bloons.arena,
            self.bloon_lookup.speed,
            self.bloon_lookup.rbe,
            self.path.waypoints,
            self.path.cumulative_distances,
            self.path.unit_directions,
            float(self.path.total_length),
        )
        if lives_lost > 0:
            self.lives = max(0, self.lives - lives_lost)
            if self.lives == 0:
                self.is_game_over = True
                self.round_active = False
                return StepResult(0, 0, lives_lost, False, True, False)

        self.round_elapsed_time += dt
        if self.round_spawn_cursor < c_round.spawn_times.shape[0]:
            self.round_spawn_cursor = jit_spawn_step(
                float(self.round_elapsed_time),
                int(self.round_spawn_cursor),
                c_round.spawn_times,
                c_round.bloon_numeric_ids,
                c_round.spawn_camo,
                self.bloons.arena,
                self.bloon_lookup.base_hp,
                self.bloon_lookup.speed,
                self.path.waypoints,
                self.path.cumulative_distances,
                self.path.unit_directions,
                float(self.path.total_length),
            )

        cash_earned, pops = jit_step_combat_v2(
            float(dt),
            self.towers.arena,
            self.bloons.arena,
            self.bloon_lookup.base_hp,
            self.bloon_lookup.rbe,
            self.bloon_lookup.cash_reward,
            self.bloon_lookup.immunity_mask,
            self.bloon_lookup.child_numeric_id,
            self.bloon_lookup.child_count,
            self.path.waypoints,
            self.path.cumulative_distances,
            self.path.unit_directions,
            float(self.path.total_length),
            self._scratch_cand,
            self._scratch_hits,
            self._scratch_queue,
        )
        self.cash += cash_earned
        self.total_pops += pops

        round_cleared = False
        spawning_done = self.round_spawn_cursor >= c_round.spawn_times.shape[0]
        if spawning_done and self.bloons.active_count == 0:
            bonus = c_round.end_of_round_cash
            self.cash += bonus
            cash_earned += bonus

            self.round_active = False
            round_cleared = True
            self.current_round_idx += 1

            if self.current_round_idx >= len(self.rounds_schedule):
                self.is_victory = True

        return StepResult(
            cash_earned=cash_earned,
            pops=pops,
            lives_lost=lives_lost,
            round_cleared=round_cleared,
            is_game_over=self.is_game_over,
            is_victory=self.is_victory,
        )

    def run_round(self, dt: float = 0.05, max_ticks: int = 36000) -> StepResult:
        if self.is_game_over or self.is_victory:
            return StepResult(0, 0, 0, False, self.is_game_over, self.is_victory)
        if self.current_round_idx >= len(self.rounds_schedule):
            self.is_victory = True
            return StepResult(0, 0, 0, False, False, True)

        c_round = self.rounds_schedule[self.current_round_idx]

        cash_earned, pops, lives_lost, ticks, cleared = jit_run_round_loop(
            float(dt),
            int(max_ticks),
            int(self.lives),
            c_round.spawn_times,
            c_round.bloon_numeric_ids,
            c_round.spawn_camo,
            self.towers.arena,
            self.bloons.arena,
            self.bloon_lookup.base_hp,
            self.bloon_lookup.speed,
            self.bloon_lookup.rbe,
            self.bloon_lookup.cash_reward,
            self.bloon_lookup.immunity_mask,
            self.bloon_lookup.child_numeric_id,
            self.bloon_lookup.child_count,
            self.path.waypoints,
            self.path.cumulative_distances,
            self.path.unit_directions,
            float(self.path.total_length),
            self._scratch_cand,
            self._scratch_hits,
            self._scratch_queue,
        )

        self.ticks_elapsed += ticks
        self.cash += cash_earned
        self.total_pops += pops
        self.lives = max(0, self.lives - lives_lost)
        self.round_active = False

        if self.lives == 0:
            self.is_game_over = True
            return StepResult(cash_earned, pops, lives_lost, False, True, False)

        if cleared:
            bonus = c_round.end_of_round_cash
            self.cash += bonus
            cash_earned += bonus
            self.current_round_idx += 1
            if self.current_round_idx >= len(self.rounds_schedule):
                self.is_victory = True

        return StepResult(
            cash_earned=cash_earned,
            pops=pops,
            lives_lost=lives_lost,
            round_cleared=cleared,
            is_game_over=self.is_game_over,
            is_victory=self.is_victory,
        )

    def get_observation_vector(self) -> npt.NDArray[np.float32]:
        """
        Vetor de estado compacto (float32) para redes neurais:
        - 5 escalares globais: [cash, lives, round_idx, active_bloons, active_towers]
        - Grid 2D achatado: (W * H,)
        - Estado de slots de torres: (max_towers * 4,) -> [active, top_t, mid_t, bot_t]
        """
        scalars = np.array(
            [
                float(self.cash),
                float(self.lives),
                float(self.current_round_idx),
                float(self.bloons.active_count),
                float(self.towers.count),
            ],
            dtype=np.float32,
        )
        flat_grid = self.grid.grid.ravel().astype(np.float32)
        tower_slots_state = np.column_stack(
            (
                self.towers.arena.active.astype(np.float32),
                self.towers.tiers.astype(np.float32),
            )
        ).ravel()
        return np.concatenate((scalars, flat_grid, tower_slots_state))
