from __future__ import annotations

from typing import Any, Dict, List, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import numpy.typing as npt

from core.arenas import TargetingMode
from core.engine import GameEngine, StepResult
from core.grid import CellState
from utils.data_loader import BloonLookupTable, CompiledRound, TowerData


class BTD6GymEnv(gym.Env[npt.NDArray[np.float32], int]):
    """
    Ambiente Gymnasium completo para PPO / MaskablePPO com suporte a:
    - Ação 0: Avançar simulação (Wave ou Tick Stride)
    - Bloco PLACE: Posicionar torre (tower_type, x, y)
    - Bloco UPGRADE: Evoluir caminho (slot, path_idx) com trava de crosspathing 5-2-0
    - Bloco SELL: Vender torre ativa (slot) recuperando 70% do investimento
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        waypoints: npt.NDArray[np.float32],
        path_cells: npt.NDArray[np.int32],
        towers_catalog: Dict[str, TowerData],
        bloon_lookup: BloonLookupTable,
        rounds_schedule: List[CompiledRound],
        max_towers: int = 64,
        round_based_decisions: bool = True,
        tick_stride: int = 20,
        dt: float = 0.05,
    ) -> None:
        super().__init__()

        self.engine = GameEngine(
            grid_width=grid_width,
            grid_height=grid_height,
            waypoints=waypoints,
            path_cells=path_cells,
            towers_catalog=towers_catalog,
            bloon_lookup=bloon_lookup,
            rounds_schedule=rounds_schedule,
            max_towers=max_towers,
        )

        self.round_based_decisions = round_based_decisions
        self.tick_stride = tick_stride
        self.dt = dt

        self.tower_ids: List[str] = sorted(towers_catalog.keys())
        self.tower_costs: npt.NDArray[np.int32] = np.array(
            [towers_catalog[t_id].base_cost for t_id in self.tower_ids],
            dtype=np.int32,
        )

        self.num_tower_types: int = len(self.tower_ids)
        self.num_cells: int = grid_width * grid_height
        self.max_towers: int = max_towers

        # Offsets do Espaço de Ações Discreto
        self.place_offset: int = 1
        self.num_place_actions: int = self.num_tower_types * self.num_cells

        self.upgrade_offset: int = self.place_offset + self.num_place_actions
        self.num_upgrade_actions: int = self.max_towers * 3

        self.sell_offset: int = self.upgrade_offset + self.num_upgrade_actions
        self.num_sell_actions: int = self.max_towers

        self.total_actions: int = self.sell_offset + self.num_sell_actions
        self.action_space = spaces.Discrete(self.total_actions)

        obs_dim = 5 + self.num_cells + (self.max_towers * 4)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _decode_place_action(self, action: int) -> Tuple[int, int, int]:
        shifted = action - self.place_offset
        tower_type_idx = shifted // self.num_cells
        cell_idx = shifted % self.num_cells
        x = cell_idx // self.engine.grid_height
        y = cell_idx % self.engine.grid_height
        return tower_type_idx, x, y

    def _decode_upgrade_action(self, action: int) -> Tuple[int, int]:
        shifted = action - self.upgrade_offset
        slot = shifted // 3
        path_idx = shifted % 3
        return slot, path_idx

    def _decode_sell_action(self, action: int) -> int:
        return action - self.sell_offset

    def action_masks(self) -> npt.NDArray[np.bool_]:
        """
        Gera a máscara booleana de ações válidas para o MaskablePPO:
        - Bloqueia compras sem saldo ou em células ocupadas
        - Bloqueia upgrades que violem crosspathing 5-2-0, tier máximo ou saldo
        - Bloqueia vendas de slots vazios
        """
        mask = np.zeros(self.total_actions, dtype=np.bool_)
        mask[0] = True  # Avançar simulação é sempre permitido

        # 1. Máscara de Posicionamento (PLACE)
        if self.engine.towers.count < self.max_towers:
            free_cells_flat = self.engine.grid.grid.ravel() == CellState.EMPTY
            if np.any(free_cells_flat):
                affordable_towers = self.tower_costs <= self.engine.cash
                valid_placements = np.logical_and(
                    affordable_towers[:, np.newaxis], free_cells_flat[np.newaxis, :]
                ).ravel()
                mask[self.place_offset : self.upgrade_offset] = valid_placements

        # 2. Máscaras de Upgrade (5-2-0) e Venda (SELL)
        active_slots = np.flatnonzero(self.engine.towers.arena.active)
        for slot_np in active_slots:
            slot = int(slot_np)
            # Venda permitida para qualquer torre ativa
            mask[self.sell_offset + slot] = True

            t_id = self.engine.towers.tower_ids[slot]
            if t_id is None:
                continue

            t_data = self.engine.towers_catalog[t_id]
            cur_tiers = (
                int(self.engine.towers.tiers[slot, 0]),
                int(self.engine.towers.tiers[slot, 1]),
                int(self.engine.towers.tiers[slot, 2]),
            )
            for path_idx in range(3):
                can_upg, cost = t_data.can_apply_next_tier(cur_tiers, path_idx)
                if can_upg and self.engine.cash >= cost:
                    mask[self.upgrade_offset + (slot * 3) + path_idx] = True

        return mask

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[npt.NDArray[np.float32], Dict[str, Any]]:
        super().reset(seed=seed)
        obs = self.engine.reset()
        return obs, self._build_info()

    def step(
        self, action: int
    ) -> Tuple[npt.NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        reward = 0.0
        action_ok = True

        # Ação 0: Avançar a simulação
        if action == 0:
            if self.round_based_decisions:
                res = self.engine.run_round(dt=self.dt)
                reward += self._compute_step_reward(res)
            else:
                if not self.engine.round_active:
                    self.engine.start_next_round()
                reward += self._run_tick_stride()

        # Bloco PLACE: Comprar e posicionar nova torre
        elif action < self.upgrade_offset:
            t_idx, x, y = self._decode_place_action(action)
            slot = self.engine.action_place_tower(
                tower_id=self.tower_ids[t_idx],
                x=x,
                y=y,
                target_mode=TargetingMode.FIRST,
            )
            action_ok = slot != -1
            if action_ok:
                reward += 0.05

        # Bloco UPGRADE: Evoluir torre existente
        elif action < self.sell_offset:
            slot, path_idx = self._decode_upgrade_action(action)
            action_ok = self.engine.action_upgrade_tower(slot, path_idx)
            if action_ok:
                reward += 0.04

        # Bloco SELL: Vender torre existente
        else:
            slot = self._decode_sell_action(action)
            refund = self.engine.action_sell_tower(slot)
            action_ok = refund >= 0
            if action_ok:
                # Leve custo de oportunidade para evitar ciclos infinitos de compra/venda
                reward -= 0.02

        if not action_ok:
            reward -= 0.1
            if self.round_based_decisions:
                res = self.engine.run_round(dt=self.dt)
                reward += self._compute_step_reward(res)

        terminated = self.engine.is_game_over or self.engine.is_victory
        truncated = False
        obs = self.engine.get_observation_vector()

        return obs, float(reward), terminated, truncated, self._build_info()

    def _run_tick_stride(self) -> float:
        stride_reward = 0.0
        for _ in range(self.tick_stride):
            if not self.engine.round_active or self.engine.is_game_over or self.engine.is_victory:
                break
            res = self.engine.step_tick(self.dt)
            stride_reward += self._compute_step_reward(res)
        return stride_reward

    @staticmethod
    def _compute_step_reward(res: StepResult) -> float:
        r = (res.pops * 0.005) - (res.lives_lost * 0.5)
        if res.round_cleared:
            r += 2.0
        if res.is_victory:
            r += 50.0
        elif res.is_game_over:
            r -= 25.0
        return r

    def _build_info(self) -> Dict[str, Any]:
        return {
            "cash": self.engine.cash,
            "lives": self.engine.lives,
            "round": self.engine.current_round_idx + 1,
            "total_pops": self.engine.total_pops,
            "towers_count": self.engine.towers.count,
        }
