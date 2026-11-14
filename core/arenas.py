from __future__ import annotations

from enum import IntEnum
from typing import List, NamedTuple, Tuple
import numpy as np
import numpy.typing as npt

from utils.data_loader import ComputedTowerStats, TowerData, TowerRole


class TargetingMode(IntEnum):
    FIRST = 0
    LAST = 1
    CLOSE = 2
    STRONG = 3


class BloonArena(NamedTuple):
    active: npt.NDArray[np.bool_]
    type_id: npt.NDArray[np.int32]
    hp: npt.NDArray[np.int32]
    is_camo: npt.NDArray[np.bool_]
    distance: npt.NDArray[np.float32]
    positions: npt.NDArray[np.float32]


class TowerArena(NamedTuple):
    active: npt.NDArray[np.bool_]
    role: npt.NDArray[np.int8]
    positions: npt.NDArray[np.float32]
    target_mode: npt.NDArray[np.int8]
    timers: npt.NDArray[np.float32]
    pop_counter: npt.NDArray[np.int64]

    # Status Base
    base_range: npt.NDArray[np.float32]
    base_damage: npt.NDArray[np.int32]
    base_pierce: npt.NDArray[np.int32]
    base_cooldown: npt.NDArray[np.float32]
    base_blast_sq: npt.NDArray[np.float32]
    base_camo: npt.NDArray[np.bool_]
    base_dmg_mask: npt.NDArray[np.int32]

    # Status Efetivos (pós-Aura estática)
    eff_range_sq: npt.NDArray[np.float32]
    eff_damage: npt.NDArray[np.int32]
    eff_pierce: npt.NDArray[np.int32]
    eff_cooldown: npt.NDArray[np.float32]
    eff_camo: npt.NDArray[np.bool_]
    eff_dmg_mask: npt.NDArray[np.int32]

    # Parâmetros de Aura emitida (Monkey Village)
    aura_range_mult: npt.NDArray[np.float32]
    aura_cd_mult: npt.NDArray[np.float32]
    aura_camo: npt.NDArray[np.bool_]

    # Parâmetros de Poção emitida (Alchemist)
    emit_buff_dur: npt.NDArray[np.float32]
    emit_buff_shots: npt.NDArray[np.int32]
    emit_buff_dmg: npt.NDArray[np.int32]
    emit_buff_pierce: npt.NDArray[np.int32]
    emit_buff_cd_mult: npt.NDArray[np.float32]
    emit_buff_lead: npt.NDArray[np.bool_]

    # Estado de Buff Dinâmico recebido pela torre
    dyn_timer: npt.NDArray[np.float32]
    dyn_shots_left: npt.NDArray[np.int32]
    dyn_dmg_bonus: npt.NDArray[np.int32]
    dyn_pierce_bonus: npt.NDArray[np.int32]
    dyn_cd_mult: npt.NDArray[np.float32]
    dyn_lead_bypass: npt.NDArray[np.bool_]


class BloonPool:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.arena = BloonArena(
            active=np.zeros(capacity, dtype=np.bool_),
            type_id=np.zeros(capacity, dtype=np.int32),
            hp=np.zeros(capacity, dtype=np.int32),
            is_camo=np.zeros(capacity, dtype=np.bool_),
            distance=np.zeros(capacity, dtype=np.float32),
            positions=np.zeros((capacity, 2), dtype=np.float32),
        )

    @property
    def active_count(self) -> int:
        return int(np.count_nonzero(self.arena.active))


class TowerManager:
    """
    Gerencia o TowerArena, rastreia tiers (top, mid, bot) e investimento total
    por slot para suportar upgrades dinâmicos (crosspathing 5-2-0) e venda.
    """

    def __init__(self, max_towers: int = 128, sell_refund_rate: float = 0.70) -> None:
        self.capacity = max_towers
        self.sell_refund_rate = sell_refund_rate
        self.count = 0

        # Metadados gerenciados fora dos arrays críticos do kernel de combate
        self.tower_ids: List[str | None] = [None] * max_towers
        self.tiers: npt.NDArray[np.int8] = np.zeros((max_towers, 3), dtype=np.int8)
        self.total_invested: npt.NDArray[np.int32] = np.zeros(max_towers, dtype=np.int32)
        self.grid_coords: npt.NDArray[np.int32] = np.full((max_towers, 2), -1, dtype=np.int32)

        self.arena = TowerArena(
            active=np.zeros(max_towers, dtype=np.bool_),
            role=np.zeros(max_towers, dtype=np.int8),
            positions=np.zeros((max_towers, 2), dtype=np.float32),
            target_mode=np.zeros(max_towers, dtype=np.int8),
            timers=np.zeros(max_towers, dtype=np.float32),
            pop_counter=np.zeros(max_towers, dtype=np.int64),
            base_range=np.zeros(max_towers, dtype=np.float32),
            base_damage=np.zeros(max_towers, dtype=np.int32),
            base_pierce=np.zeros(max_towers, dtype=np.int32),
            base_cooldown=np.zeros(max_towers, dtype=np.float32),
            base_blast_sq=np.zeros(max_towers, dtype=np.float32),
            base_camo=np.zeros(max_towers, dtype=np.bool_),
            base_dmg_mask=np.zeros(max_towers, dtype=np.int32),
            eff_range_sq=np.zeros(max_towers, dtype=np.float32),
            eff_damage=np.zeros(max_towers, dtype=np.int32),
            eff_pierce=np.zeros(max_towers, dtype=np.int32),
            eff_cooldown=np.zeros(max_towers, dtype=np.float32),
            eff_camo=np.zeros(max_towers, dtype=np.bool_),
            eff_dmg_mask=np.zeros(max_towers, dtype=np.int32),
            aura_range_mult=np.ones(max_towers, dtype=np.float32),
            aura_cd_mult=np.ones(max_towers, dtype=np.float32),
            aura_camo=np.zeros(max_towers, dtype=np.bool_),
            emit_buff_dur=np.zeros(max_towers, dtype=np.float32),
            emit_buff_shots=np.zeros(max_towers, dtype=np.int32),
            emit_buff_dmg=np.zeros(max_towers, dtype=np.int32),
            emit_buff_pierce=np.zeros(max_towers, dtype=np.int32),
            emit_buff_cd_mult=np.ones(max_towers, dtype=np.float32),
            emit_buff_lead=np.zeros(max_towers, dtype=np.bool_),
            dyn_timer=np.zeros(max_towers, dtype=np.float32),
            dyn_shots_left=np.zeros(max_towers, dtype=np.int32),
            dyn_dmg_bonus=np.zeros(max_towers, dtype=np.int32),
            dyn_pierce_bonus=np.zeros(max_towers, dtype=np.int32),
            dyn_cd_mult=np.ones(max_towers, dtype=np.float32),
            dyn_lead_bypass=np.zeros(max_towers, dtype=np.bool_),
        )

    def _write_stats_to_slot(self, slot: int, stats: ComputedTowerStats) -> None:
        a = self.arena
        a.role[slot] = int(stats.role)
        a.base_range[slot] = stats.range_grid
        a.base_damage[slot] = stats.damage
        a.base_pierce[slot] = stats.pierce
        a.base_cooldown[slot] = stats.cooldown
        a.base_blast_sq[slot] = stats.blast_radius ** 2
        a.base_camo[slot] = stats.can_see_camo
        a.base_dmg_mask[slot] = stats.damage_mask

        a.aura_range_mult[slot] = stats.aura_range_mult
        a.aura_cd_mult[slot] = stats.aura_cooldown_mult
        a.aura_camo[slot] = stats.aura_grants_camo

        a.emit_buff_dur[slot] = stats.buff_duration
        a.emit_buff_shots[slot] = stats.buff_shots
        a.emit_buff_dmg[slot] = stats.buff_damage_bonus
        a.emit_buff_pierce[slot] = stats.buff_pierce_bonus
        a.emit_buff_cd_mult[slot] = stats.buff_cooldown_mult
        a.emit_buff_lead[slot] = stats.buff_grants_lead_pop

    def add_configured_tower(
        self,
        tower_id: str,
        x: int,
        y: int,
        stats: ComputedTowerStats,
        tiers: Tuple[int, int, int] = (0, 0, 0),
        target_mode: TargetingMode = TargetingMode.FIRST,
    ) -> int:
        inactive = np.flatnonzero(~self.arena.active)
        if inactive.size == 0:
            return -1

        slot = int(inactive[0])
        a = self.arena

        self.tower_ids[slot] = tower_id
        self.tiers[slot] = tiers
        self.total_invested[slot] = stats.total_cost
        self.grid_coords[slot] = (x, y)

        a.active[slot] = True
        a.positions[slot] = (np.float32(x), np.float32(y))
        a.target_mode[slot] = int(target_mode)
        a.timers[slot] = 0.0
        a.pop_counter[slot] = 0
        a.dyn_timer[slot] = 0.0
        a.dyn_shots_left[slot] = 0

        self._write_stats_to_slot(slot, stats)
        self.count += 1
        self.recompute_static_auras()
        return slot

    def upgrade_tower(
        self, slot: int, path_idx: int, tower_data: TowerData, available_cash: int
    ) -> int:
        """
        Tenta aplicar o próximo tier do caminho `path_idx` na torre do `slot`.
        Retorna o custo da melhoria se bem-sucedido, ou -1 se inválido/sem saldo.
        """
        if not (0 <= slot < self.capacity and self.arena.active[slot]):
            return -1

        cur_t = (
            int(self.tiers[slot, 0]),
            int(self.tiers[slot, 1]),
            int(self.tiers[slot, 2]),
        )
        can_upg, cost = tower_data.can_apply_next_tier(cur_t, path_idx)
        if not can_upg or available_cash < cost:
            return -1

        new_tiers = [cur_t[0], cur_t[1], cur_t[2]]
        new_tiers[path_idx] += 1
        new_stats = tower_data.compute_full_stats(
            new_tiers[0], new_tiers[1], new_tiers[2]
        )

        self.tiers[slot] = new_tiers
        self.total_invested[slot] = new_stats.total_cost
        self._write_stats_to_slot(slot, new_stats)
        self.recompute_static_auras()
        return cost

    def sell_tower(self, slot: int) -> Tuple[int, int, int]:
        """
        Vende a torre ativa no `slot`, recalcula auras estáticas e retorna
        (valor_reembolsado, grid_x, grid_y). Retorna (-1, -1, -1) se inválido.
        """
        if not (0 <= slot < self.capacity and self.arena.active[slot]):
            return -1, -1, -1

        refund = int(self.total_invested[slot] * self.sell_refund_rate)
        gx = int(self.grid_coords[slot, 0])
        gy = int(self.grid_coords[slot, 1])

        a = self.arena
        a.active[slot] = False
        a.dyn_timer[slot] = 0.0
        a.dyn_shots_left[slot] = 0

        self.tower_ids[slot] = None
        self.tiers[slot] = (0, 0, 0)
        self.total_invested[slot] = 0
        self.grid_coords[slot] = (-1, -1)

        self.count -= 1
        self.recompute_static_auras()
        return refund, gx, gy

    def recompute_static_auras(self) -> None:
        a = self.arena
        active_idx = np.flatnonzero(a.active)
        if active_idx.size == 0:
            return

        a.eff_range_sq[active_idx] = a.base_range[active_idx] ** 2
        a.eff_damage[active_idx] = a.base_damage[active_idx]
        a.eff_pierce[active_idx] = a.base_pierce[active_idx]
        a.eff_cooldown[active_idx] = a.base_cooldown[active_idx]
        a.eff_camo[active_idx] = a.base_camo[active_idx]
        a.eff_dmg_mask[active_idx] = a.base_dmg_mask[active_idx]

        villages = active_idx[a.role[active_idx] == int(TowerRole.VILLAGE_AURA)]
        if villages.size == 0:
            return

        for t_idx in active_idx:
            if a.role[t_idx] == int(TowerRole.VILLAGE_AURA):
                continue

            best_range_mult = 1.0
            best_cd_mult = 1.0
            granted_camo = bool(a.base_camo[t_idx])
            t_pos = a.positions[t_idx]

            for v_idx in villages:
                v_pos = a.positions[v_idx]
                dist_sq = float(
                    (t_pos[0] - v_pos[0]) ** 2 + (t_pos[1] - v_pos[1]) ** 2
                )
                if dist_sq <= a.eff_range_sq[v_idx]:
                    if a.aura_range_mult[v_idx] > best_range_mult:
                        best_range_mult = float(a.aura_range_mult[v_idx])
                    if a.aura_cd_mult[v_idx] < best_cd_mult:
                        best_cd_mult = float(a.aura_cd_mult[v_idx])
                    if a.aura_camo[v_idx]:
                        granted_camo = True

            eff_r = float(a.base_range[t_idx]) * best_range_mult
            a.eff_range_sq[t_idx] = np.float32(eff_r * eff_r)
            a.eff_cooldown[t_idx] = np.float32(float(a.base_cooldown[t_idx]) * best_cd_mult)
            a.eff_camo[t_idx] = granted_camo
