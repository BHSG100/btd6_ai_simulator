from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag
from pathlib import Path
from typing import Dict, List, Literal, Tuple
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TowerRole(IntEnum):
    ATTACKER = 0
    VILLAGE_AURA = 1
    ALCHEMIST_BUFFER = 2
    ECONOMY = 3  # Adicionado para suportar Banana Farm


ROLE_MAP: Dict[str, TowerRole] = {
    "attacker": TowerRole.ATTACKER,
    "village_aura": TowerRole.VILLAGE_AURA,
    "alchemist_buffer": TowerRole.ALCHEMIST_BUFFER,
    "economy": TowerRole.ECONOMY,  # Mapeamento do novo role
}

class DamageImmunity(IntFlag):
    NONE = 0
    SHARP = 1 << 0
    EXPLOSION = 1 << 1
    FREEZE = 1 << 2
    ENERGY = 1 << 3


IMMUNITY_MAP: Dict[str, DamageImmunity] = {
    "sharp": DamageImmunity.SHARP,
    "explosion": DamageImmunity.EXPLOSION,
    "freeze": DamageImmunity.FREEZE,
    "energy": DamageImmunity.ENERGY,
}

DAMAGE_TYPE_MAP: Dict[str, DamageImmunity] = {
    "none": DamageImmunity.NONE,
    **IMMUNITY_MAP,
}


def is_valid_crosspath(top_tier: int, mid_tier: int, bot_tier: int) -> bool:
    """
    Valida a regra oficial de crosspathing do BTD6 (máximo 5-2-0):
    1. Tiers devem estar entre 0 e 5.
    2. No máximo 2 caminhos podem ter tier > 0.
    3. No máximo 1 caminho pode ter tier >= 3.
    """
    tiers = (top_tier, mid_tier, bot_tier)
    if any(t < 0 or t > 5 for t in tiers):
        return False
    active_paths = sum(1 for t in tiers if t > 0)
    if active_paths > 2:
        return False
    major_paths = sum(1 for t in tiers if t >= 3)
    if major_paths > 1:
        return False
    return True


class UpgradeTier(BaseModel):
    model_config = ConfigDict(frozen=True)

    tier: int = Field(ge=1, le=5)
    id: str
    name: str
    cost: int = Field(ge=0)
    damage_delta: int = 0
    pierce_delta: int = 0
    range_delta: float = 0.0
    blast_radius_delta: float = 0.0
    cooldown_multiplier: float = Field(default=1.0, gt=0.0)
    grants_camo: bool = False
    aura_range_mult_delta: float = 0.0
    aura_cooldown_mult_delta: float = 0.0
    aura_grants_camo: bool = False
    buff_duration: float = 0.0
    buff_shots: int = 0
    buff_damage_bonus: int = 0
    buff_pierce_bonus: int = 0
    buff_cooldown_mult: float = 1.0
    buff_grants_lead_pop: bool = False


class UpgradePaths(BaseModel):
    model_config = ConfigDict(frozen=True)

    top_path: List[UpgradeTier] = Field(default_factory=list)
    middle_path: List[UpgradeTier] = Field(default_factory=list)
    bottom_path: List[UpgradeTier] = Field(default_factory=list)

    def get_path_list(self, path_idx: int) -> List[UpgradeTier]:
        if path_idx == 0:
            return self.top_path
        if path_idx == 1:
            return self.middle_path
        if path_idx == 2:
            return self.bottom_path
        raise IndexError(f"path_idx inválido: {path_idx}. Esperado 0, 1 ou 2.")


@dataclass(frozen=True)
class ComputedTowerStats:
    """Estrutura tipada com todos os status consolidados da torre e seus upgrades."""
    total_cost: int
    role: TowerRole
    range_grid: float
    damage: int
    pierce: int
    cooldown: float
    blast_radius: float
    can_see_camo: bool
    damage_mask: int
    aura_range_mult: float
    aura_cooldown_mult: float
    aura_grants_camo: bool
    buff_duration: float
    buff_shots: int
    buff_damage_bonus: int
    buff_pierce_bonus: int
    buff_cooldown_mult: float
    buff_grants_lead_pop: bool


class TowerData(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    role: Literal["attacker", "village_aura", "alchemist_buffer", "economy"] = "attacker"  # Adicionado "economy"
    base_cost: int = Field(gt=0)
    range_grid: float = Field(gt=0.0)
    base_damage: int = Field(ge=0)
    base_pierce: int = Field(ge=0)  # Alterado de ge=1 para ge=0 (suporta farm com pierce 0)
    cooldown_seconds: float = Field(gt=0.0)
    blast_radius: float = Field(default=0.0, ge=0.0)
    can_see_camo: bool = False
    damage_type: Literal["none", "sharp", "explosion", "freeze", "energy", "glue"] = "sharp"  # Adicionado "glue" se necessário
    footprint_radius: float = Field(gt=0.0)
    aura_range_mult: float = 1.0
    aura_cooldown_mult: float = 1.0
    aura_grants_camo: bool = False
    upgrades: UpgradePaths

    def can_apply_next_tier(
        self, current_tiers: Tuple[int, int, int], path_idx: int
    ) -> Tuple[bool, int]:
        """
        Verifica se a torre pode evoluir o próximo tier do caminho `path_idx`
        respeitando os tiers disponíveis no JSON e a trava de crosspathing 5-2-0.
        Retorna: (is_valid, next_tier_cost).
        """
        if path_idx not in (0, 1, 2):
            return False, 0

        path_list = self.upgrades.get_path_list(path_idx)
        cur_t = current_tiers[path_idx]
        if cur_t >= len(path_list):
            return False, 0

        next_tiers = list(current_tiers)
        next_tiers[path_idx] = cur_t + 1
        if not is_valid_crosspath(next_tiers[0], next_tiers[1], next_tiers[2]):
            return False, 0

        return True, path_list[cur_t].cost

    def compute_full_stats(
        self, top_tier: int = 0, mid_tier: int = 0, bot_tier: int = 0
    ) -> ComputedTowerStats:
        if not is_valid_crosspath(top_tier, mid_tier, bot_tier):
            raise ValueError(
                f"Combinação de crosspathing inválida ({top_tier}-{mid_tier}-{bot_tier})."
            )
        if (
            top_tier > len(self.upgrades.top_path)
            or mid_tier > len(self.upgrades.middle_path)
            or bot_tier > len(self.upgrades.bottom_path)
        ):
            raise ValueError(
                f"Tier solicitado excede os upgrades definidos para '{self.id}'."
            )

        cost = self.base_cost
        rng = self.range_grid
        dmg = self.base_damage
        prc = self.base_pierce
        cd = self.cooldown_seconds
        blast = self.blast_radius
        camo = self.can_see_camo
        dmg_mask = int(DAMAGE_TYPE_MAP[self.damage_type])

        a_rng = self.aura_range_mult
        a_cd = self.aura_cooldown_mult
        a_camo = self.aura_grants_camo

        b_dur = 0.0
        b_shots = 0
        b_dmg = 0
        b_prc = 0
        b_cd = 1.0
        b_lead = False

        active_upgs = (
            list(self.upgrades.top_path[:top_tier])
            + list(self.upgrades.middle_path[:mid_tier])
            + list(self.upgrades.bottom_path[:bot_tier])
        )

        for u in active_upgs:
            cost += u.cost
            rng += u.range_delta
            dmg += u.damage_delta
            prc += u.pierce_delta
            cd *= u.cooldown_multiplier
            blast += u.blast_radius_delta
            camo = camo or u.grants_camo
            a_rng += u.aura_range_mult_delta
            a_cd += u.aura_cooldown_mult_delta
            a_camo = a_camo or u.aura_grants_camo
            if u.buff_shots > 0 or u.buff_duration > 0.0:
                b_dur = max(b_dur, u.buff_duration)
                b_shots = max(b_shots, u.buff_shots)
                b_dmg += u.buff_damage_bonus
                b_prc += u.buff_pierce_bonus
                b_cd *= u.buff_cooldown_mult
                b_lead = b_lead or u.buff_grants_lead_pop

        return ComputedTowerStats(
            total_cost=cost,
            role=ROLE_MAP[self.role],
            range_grid=rng,
            damage=dmg,
            pierce=prc,
            cooldown=cd,
            blast_radius=blast,
            can_see_camo=camo,
            damage_mask=dmg_mask,
            aura_range_mult=a_rng,
            aura_cooldown_mult=a_cd,
            aura_grants_camo=a_camo,
            buff_duration=b_dur,
            buff_shots=b_shots,
            buff_damage_bonus=b_dmg,
            buff_pierce_bonus=b_prc,
            buff_cooldown_mult=b_cd,
            buff_grants_lead_pop=b_lead,
        )


class TowersDatabase(BaseModel):
    model_config = ConfigDict(frozen=True)

    towers: Dict[str, TowerData]


class ChildSpawn(BaseModel):
    model_config = ConfigDict(frozen=True)

    bloon_id: str
    count: int = Field(ge=1)


class BloonData(BaseModel):
    model_config = ConfigDict(frozen=True)

    numeric_id: int = Field(ge=0)
    id: str
    base_hp: int = Field(ge=1)
    speed: float = Field(gt=0.0)
    rbe: int = Field(ge=1)
    cash_reward: int = Field(ge=0)
    is_moab_class: bool = False
    immunities: List[Literal["sharp", "explosion", "freeze", "energy"]] = Field(
        default_factory=list
    )
    children: List[ChildSpawn] = Field(default_factory=list)

    @property
    def immunity_mask(self) -> int:
        mask = int(DamageImmunity.NONE)
        for imm in self.immunities:
            mask |= int(IMMUNITY_MAP[imm])
        return mask


class BloonsDatabase(BaseModel):
    model_config = ConfigDict(frozen=True)

    bloons: Dict[str, BloonData]


@dataclass(frozen=True)
class BloonLookupTable:
    base_hp: npt.NDArray[np.int32]
    speed: npt.NDArray[np.float32]
    rbe: npt.NDArray[np.int32]
    cash_reward: npt.NDArray[np.int32]
    immunity_mask: npt.NDArray[np.int32]
    child_numeric_id: npt.NDArray[np.int32]
    child_count: npt.NDArray[np.int32]


class BloonSpawnGroup(BaseModel):
    model_config = ConfigDict(frozen=True)

    bloon_id: str
    count: int = Field(ge=1)
    start_time: float = Field(ge=0.0)
    end_time: float = Field(ge=0.0)
    camo: bool = False

    @model_validator(mode="after")
    def validate_time_window(self) -> BloonSpawnGroup:
        if self.end_time < self.start_time:
            raise ValueError("end_time deve ser maior ou igual a start_time.")
        return self


class RoundConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    round_number: int = Field(ge=1)
    end_of_round_cash: int = Field(ge=0)
    groups: List[BloonSpawnGroup] = Field(min_length=1)


class RoundsDatabase(BaseModel):
    model_config = ConfigDict(frozen=True)

    rounds: List[RoundConfig]


@dataclass(frozen=True)
class CompiledRound:
    round_number: int
    end_of_round_cash: int
    spawn_times: npt.NDArray[np.float32]
    bloon_numeric_ids: npt.NDArray[np.int32]
    spawn_camo: npt.NDArray[np.bool_]


class GameDataLoader:
    @staticmethod
    def load_towers(json_path: str | Path) -> Dict[str, TowerData]:
        path = Path(json_path)
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo de torres não encontrado: {path}")
        return TowersDatabase.model_validate_json(path.read_bytes()).towers

    @staticmethod
    def load_bloons(json_path: str | Path) -> Tuple[Dict[str, BloonData], BloonLookupTable]:
        path = Path(json_path)
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo de balões não encontrado: {path}")

        validated = BloonsDatabase.model_validate_json(path.read_bytes()).bloons
        num_types = max(b.numeric_id for b in validated.values()) + 1

        base_hp = np.zeros(num_types, dtype=np.int32)
        speed = np.zeros(num_types, dtype=np.float32)
        rbe = np.zeros(num_types, dtype=np.int32)
        cash = np.zeros(num_types, dtype=np.int32)
        immunities = np.zeros(num_types, dtype=np.int32)
        child_id = np.full(num_types, -1, dtype=np.int32)
        child_cnt = np.zeros(num_types, dtype=np.int32)

        for b in validated.values():
            idx = b.numeric_id
            base_hp[idx] = b.base_hp
            speed[idx] = b.speed
            rbe[idx] = b.rbe
            cash[idx] = b.cash_reward
            immunities[idx] = b.immunity_mask
            if b.children:
                first_child = b.children[0]
                child_id[idx] = validated[first_child.bloon_id].numeric_id
                child_cnt[idx] = first_child.count

        lookup = BloonLookupTable(
            base_hp=base_hp,
            speed=speed,
            rbe=rbe,
            cash_reward=cash,
            immunity_mask=immunities,
            child_numeric_id=child_id,
            child_count=child_cnt,
        )
        return validated, lookup

    @staticmethod
    def load_and_compile_rounds(
        json_path: str | Path, bloons_db: Dict[str, BloonData]
    ) -> List[CompiledRound]:
        path = Path(json_path)
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo de rounds não encontrado: {path}")

        validated = RoundsDatabase.model_validate_json(path.read_bytes()).rounds
        compiled_list: List[CompiledRound] = []

        for r_cfg in validated:
            times_chunks: List[npt.NDArray[np.float32]] = []
            ids_chunks: List[npt.NDArray[np.int32]] = []
            camo_chunks: List[npt.NDArray[np.bool_]] = []

            for grp in r_cfg.groups:
                if grp.bloon_id not in bloons_db:
                    raise KeyError(f"Bloon '{grp.bloon_id}' não existe em bloons.json.")

                num_id = bloons_db[grp.bloon_id].numeric_id
                if grp.count == 1:
                    t_arr = np.array([grp.start_time], dtype=np.float32)
                else:
                    t_arr = np.linspace(
                        grp.start_time, grp.end_time, num=grp.count, dtype=np.float32
                    )

                id_arr = np.full(grp.count, num_id, dtype=np.int32)
                camo_arr = np.full(grp.count, grp.camo, dtype=np.bool_)

                times_chunks.append(t_arr)
                ids_chunks.append(id_arr)
                camo_chunks.append(camo_arr)

            all_times = np.concatenate(times_chunks)
            all_ids = np.concatenate(ids_chunks)
            all_camo = np.concatenate(camo_chunks)

            sort_order = np.argsort(all_times, kind="mergesort")
            compiled_list.append(
                CompiledRound(
                    round_number=r_cfg.round_number,
                    end_of_round_cash=r_cfg.end_of_round_cash,
                    spawn_times=all_times[sort_order],
                    bloon_numeric_ids=all_ids[sort_order],
                    spawn_camo=all_camo[sort_order],
                )
            )

        return compiled_list
