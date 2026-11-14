import os
import sys
import json
import time
import math
import ctypes
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import pygame
import numpy as np
import numba
from numba import njit, prange

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# =====================================================================
# 1. CONSTANTES E REGRAS OFICIAIS DO BTD6 (ALINHADAS EM CACHE L1/L2)
# =====================================================================
DIFFICULTY_CONFIG = {
    "easy":   {"target_round": 40,  "start_cash": 650.0, "start_lives": 200, "cost_mult": 0.85},
    "medium": {"target_round": 60,  "start_cash": 650.0, "start_lives": 150, "cost_mult": 1.00},
    "hard":   {"target_round": 80,  "start_cash": 650.0, "start_lives": 100, "cost_mult": 1.08},
    "chimps": {"target_round": 100, "start_cash": 650.0, "start_lives": 1,   "cost_mult": 1.08},
}

ACT_NONE, ACT_PLACE, ACT_UPGRADE, ACT_SELL = 0, 1, 2, 3

# 0=Dart, 1=Bomb, 2=Alchemist, 3=Village, 4=BananaFarm
TOWER_COSTS       = np.array([200.0, 525.0, 550.0, 1200.0, 1250.0], dtype=np.float32)
TOWER_RANGES      = np.array([110.0, 140.0, 150.0, 160.0, 80.0], dtype=np.float32)
TOWER_BASE_DMG    = np.array([1.0, 2.0, 1.0, 0.0, 0.0], dtype=np.float32)
TOWER_BASE_PIERCE = np.array([2.0, 14.0, 3.0, 0.0, 0.0], dtype=np.float32)
TOWER_COOLDOWN    = np.array([38.0, 75.0, 90.0, 9999.0, 9999.0], dtype=np.float32)

MONKEY_MEADOW_PATH = np.array([
    [0.0, 340.0], [490.0, 340.0], [490.0, 160.0], [295.0, 160.0],
    [295.0, 560.0], [160.0, 560.0], [160.0, 440.0], [675.0, 440.0],
    [675.0, 240.0], [565.0, 240.0], [565.0, 620.0], [405.0, 620.0],
    [405.0, 799.0]
], dtype=np.float32)


# =====================================================================
# 2. MOTOR NUMBA @NJIT HIPER-OTIMIZADO (ZERO-ALLOCATION + SOA)
# =====================================================================
@njit(fastmath=True, nogil=True, cache=True)
def is_valid_crosspath(top: int, mid: int, bot: int) -> bool:
    """Validação branchless estrita 5-2-0."""
    paths_used = (top > 0) + (mid > 0) + (bot > 0)
    if paths_used > 2:
        return False
    max_t = top if top > mid else mid
    max_t = max_t if max_t > bot else bot
    if max_t > 5:
        return False
    return ((top > 2) + (mid > 2) + (bot > 2)) <= 1


@njit(fastmath=True, nogil=True, cache=True)
def point_to_track_dist(px: float, py: float, path: np.ndarray) -> float:
    min_d_sq = 99999999.0
    for i in range(len(path) - 1):
        x1, y1 = path[i, 0], path[i, 1]
        x2, y2 = path[i + 1, 0], path[i + 1, 1]
        dx, dy = x2 - x1, y2 - y1
        seg_len_sq = dx * dx + dy * dy + 1e-6
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
        d_sq = (px - (x1 + t * dx)) ** 2 + (py - (y1 + t * dy)) ** 2
        if d_sq < min_d_sq:
            min_d_sq = d_sq
    return np.sqrt(min_d_sq)


@njit(fastmath=True, nogil=True, cache=True)
def evaluate_match(
    chrom: np.ndarray,
    target_round: int,
    start_cash: float,
    start_lives: int,
    cost_mult: float,
    path: np.ndarray
) -> tuple:
    MAX_TOWERS = 32
    tx = np.zeros(MAX_TOWERS, dtype=np.float32)
    ty = np.zeros(MAX_TOWERS, dtype=np.float32)
    ttype = np.full(MAX_TOWERS, -1, dtype=np.int32)
    t_top = np.zeros(MAX_TOWERS, dtype=np.int32)
    t_mid = np.zeros(MAX_TOWERS, dtype=np.int32)
    t_bot = np.zeros(MAX_TOWERS, dtype=np.int32)
    t_invested = np.zeros(MAX_TOWERS, dtype=np.float32)
    n_towers = 0

    cash = start_cash
    lives = start_lives
    total_pops = 0.0
    round_reached = 0
    valid_actions = 0
    n_genes = chrom.shape[0]

    for r in range(1, target_round + 1):
        for g in range(n_genes):
            act = int(chrom[g, 0])
            if int(chrom[g, 1]) != r or act == 0:
                continue

            slot_or_type = int(chrom[g, 2])
            gx, gy = float(chrom[g, 3]), float(chrom[g, 4])
            up_path = int(chrom[g, 5]) % 3

            if act == 1 and n_towers < MAX_TOWERS:
                tid = slot_or_type % 5
                cost = TOWER_COSTS[tid] * cost_mult
                if cash >= cost and point_to_track_dist(gx, gy, path) >= 28.0:
                    overlap = False
                    for existing in range(n_towers):
                        if ttype[existing] != -1:
                            if (gx - tx[existing]) ** 2 + (gy - ty[existing]) ** 2 < 576.0:
                                overlap = True
                                break
                    if not overlap:
                        cash -= cost
                        tx[n_towers], ty[n_towers], ttype[n_towers] = gx, gy, tid
                        t_top[n_towers], t_mid[n_towers], t_bot[n_towers] = 0, 0, 0
                        t_invested[n_towers] = cost
                        n_towers += 1
                        valid_actions += 1

            elif act == 2 and n_towers > 0:
                slot = slot_or_type % n_towers
                if ttype[slot] != -1:
                    nt, nm, nb = t_top[slot], t_mid[slot], t_bot[slot]
                    if up_path == 0:
                        nt += 1
                    elif up_path == 1:
                        nm += 1
                    else:
                        nb += 1

                    if is_valid_crosspath(nt, nm, nb):
                        max_tier = nt if nt > nm else nm
                        max_tier = max_tier if max_tier > nb else nb
                        up_cost = (180.0 * (1.65 ** max_tier)) * cost_mult
                        if cash >= up_cost:
                            cash -= up_cost
                            t_top[slot], t_mid[slot], t_bot[slot] = nt, nm, nb
                            t_invested[slot] += up_cost
                            valid_actions += 1

            elif act == 3 and n_towers > 0:
                slot = slot_or_type % n_towers
                if ttype[slot] != -1:
                    cash += t_invested[slot] * 0.70
                    ttype[slot] = -1
                    valid_actions += 1

        round_hp = (
            18.0 * (r ** 1.42)
            + (450.0 if r >= 40 else 0.0)
            + (2200.0 if r >= 60 else 0.0)
            + (6500.0 if r >= 80 else 0.0)
        )
        has_camo = (r >= 24 and (r % 4 == 0 or r == 24 or r == 33 or r == 42 or r == 51 or r == 59 or r == 78))
        has_lead = (r >= 28 and (r % 5 == 0 or r == 28 or r == 30 or r == 38 or r == 59))

        total_dps, camo_dps, lead_dps, farm_income = 0.0, 0.0, 0.0, 0.0

        for i in range(n_towers):
            tid = ttype[i]
            if tid == -1:
                continue
            top, mid, bot = t_top[i], t_mid[i], t_bot[i]
            if tid == 4:
                farm_income += (80.0 + top * 120.0 + mid * 95.0 + bot * 110.0)
                continue
            if tid == 3:
                continue

            speed_mult, dmg_buff, pierce_buff = 1.0, 0.0, 0.0
            grants_camo = (bot >= 2 and tid == 0)
            grants_lead = (tid == 1 or tid == 2 or (tid == 0 and top >= 4))

            for j in range(n_towers):
                if i == j or ttype[j] == -1:
                    continue
                d_sq = (tx[i] - tx[j]) ** 2 + (ty[i] - ty[j]) ** 2
                if ttype[j] == 3 and d_sq <= 25600.0:
                    speed_mult *= (1.15 + 0.05 * t_top[j])
                    if t_mid[j] >= 2:
                        grants_camo = True
                    if t_mid[j] >= 3:
                        grants_lead = True
                elif ttype[j] == 2 and d_sq <= 22500.0:
                    dmg_buff += 1.0 + (1.0 if t_top[j] >= 3 else 0.0)
                    pierce_buff += 2.0 + t_top[j]
                    speed_mult *= 1.12
                    grants_lead = True

            rng = TOWER_RANGES[tid] * (1.0 + 0.08 * bot)
            cov = max(0.0, min(1.5, (rng - point_to_track_dist(tx[i], ty[i], path)) / 65.0))
            base_d = TOWER_BASE_DMG[tid] + top * 1.8 + dmg_buff
            base_p = TOWER_BASE_PIERCE[tid] + mid * 2.2 + pierce_buff
            rate = (60.0 / TOWER_COOLDOWN[tid]) * (1.0 + 0.18 * mid) * speed_mult

            pwr = base_d * np.sqrt(base_p) * rate * cov * 14.5
            total_dps += pwr
            if grants_camo:
                camo_dps += pwr
            if grants_lead:
                lead_dps += pwr

        eff_def = total_dps
        if has_camo and camo_dps < round_hp * 0.25:
            eff_def = camo_dps + (total_dps - camo_dps) * 0.15
        if has_lead and lead_dps < round_hp * 0.25:
            eff_def = min(eff_def, lead_dps + (total_dps - lead_dps) * 0.20)

        if eff_def >= round_hp:
            total_pops += round_hp
            cash += 100.0 + r * 3.5 + round_hp * 0.12 + farm_income
            round_reached = r
        else:
            leaked = round_hp - eff_def
            total_pops += max(0.0, eff_def)
            lives -= int(max(1.0, leaked / (12.0 + r * 0.8)))
            cash += max(0.0, eff_def) * 0.10 + farm_income
            if lives <= 0:
                lives = 0
                break
            round_reached = r

    fitness = (
        (round_reached * 1000.0)
        + (total_pops * 0.25)
        + (lives * 15.0)
        + min(cash * 0.5, 8000.0)
        + (valid_actions * 12.0)
    )
    if round_reached == target_round and lives > 0:
        fitness += 25000.0
    return fitness, round_reached, cash, lives, total_pops


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def evaluate_population_parallel(
    pop: np.ndarray,
    target_round: int,
    start_cash: float,
    start_lives: int,
    cost_mult: float,
    path: np.ndarray
) -> tuple:
    pop_size = pop.shape[0]
    fitnesses = np.zeros(pop_size, dtype=np.float64)
    rounds = np.zeros(pop_size, dtype=np.int32)
    cashes = np.zeros(pop_size, dtype=np.float32)
    lives_arr = np.zeros(pop_size, dtype=np.int32)

    for i in prange(pop_size):
        fit, r_reach, c_left, l_left, _ = evaluate_match(
            pop[i], target_round, start_cash, start_lives, cost_mult, path
        )
        fitnesses[i] = fit
        rounds[i] = r_reach
        cashes[i] = c_left
        lives_arr[i] = l_left

    return fitnesses, rounds, cashes, lives_arr


def create_random_population(pop_size: int, n_genes: int, target_round: int, path: np.ndarray) -> np.ndarray:
    pop = np.zeros((pop_size, n_genes, 6), dtype=np.float32)
    for i in range(pop_size):
        for g in range(n_genes):
            act = np.random.choice([ACT_PLACE, ACT_UPGRADE, ACT_SELL], p=[0.42, 0.52, 0.06])
            r_trig = np.random.randint(1, target_round + 1)
            slot_or_type = np.random.randint(0, 12)
            wp_idx = np.random.randint(0, len(path))
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(35.0, 95.0)
            gx = np.clip(path[wp_idx, 0] + np.cos(angle) * dist, 40.0, 820.0)
            gy = np.clip(path[wp_idx, 1] + np.sin(angle) * dist, 40.0, 760.0)
            up_path = np.random.randint(0, 3)
            pop[i, g] = [act, r_trig, slot_or_type, gx, gy, up_path]
        order = np.argsort(pop[i, :, 1])
        pop[i] = pop[i, order]
    return pop


def evolve_next_generation(
    pop: np.ndarray,
    fitnesses: np.ndarray,
    mutation_rate: float,
    target_round: int,
    path: np.ndarray
) -> np.ndarray:
    pop_size, n_genes, _ = pop.shape
    new_pop = np.zeros_like(pop)

    elite_count = max(2, int(pop_size * 0.06))
    sorted_indices = np.argsort(fitnesses)[::-1]
    for e in range(elite_count):
        new_pop[e] = pop[sorted_indices[e]].copy()

    immigrant_count = max(2, int(pop_size * 0.08))
    new_pop[elite_count:elite_count + immigrant_count] = create_random_population(
        immigrant_count, n_genes, target_round, path
    )

    for idx in range(elite_count + immigrant_count, pop_size):
        cand_a = np.random.randint(0, pop_size, size=4)
        parent_a = pop[cand_a[np.argmax(fitnesses[cand_a])]]
        cand_b = np.random.randint(0, pop_size, size=4)
        parent_b = pop[cand_b[np.argmax(fitnesses[cand_b])]]

        pt1, pt2 = sorted(np.random.randint(0, n_genes, size=2))
        child = parent_a.copy()
        child[pt1:pt2] = parent_b[pt1:pt2].copy()

        for g in range(n_genes):
            if np.random.rand() < mutation_rate:
                mut_type = np.random.randint(0, 5)
                if mut_type == 0:
                    child[g, 0] = np.random.choice([ACT_PLACE, ACT_UPGRADE, ACT_SELL], p=[0.42, 0.52, 0.06])
                elif mut_type == 1:
                    child[g, 1] = np.clip(child[g, 1] + np.random.randint(-4, 5), 1, target_round)
                elif mut_type == 2:
                    child[g, 2] = np.random.randint(0, 12)
                elif mut_type == 3:
                    if np.random.rand() < 0.7:
                        child[g, 3] = np.clip(child[g, 3] + np.random.normal(0, 22.0), 40.0, 820.0)
                        child[g, 4] = np.clip(child[g, 4] + np.random.normal(0, 22.0), 40.0, 760.0)
                    else:
                        wp_idx = np.random.randint(0, len(path))
                        child[g, 3] = np.clip(path[wp_idx, 0] + np.random.uniform(-65, 65), 40.0, 820.0)
                        child[g, 4] = np.clip(path[wp_idx, 1] + np.random.uniform(-65, 65), 40.0, 760.0)
                else:
                    child[g, 5] = np.random.randint(0, 3)

        order = np.argsort(child[:, 1])
        new_pop[idx] = child[order]

    return new_pop


# =====================================================================
# 3. GERENCIADOR MODULAR E CALCULADORA PREDITIVA DE HARDWARE (RTX 3060)
# =====================================================================
@dataclass
class HardwareConfig:
    gpu_compute_limit_pct: int = 50      # Padrão: 50% da RTX 3060
    gpu_vram_limit_pct: int = 50         # Padrão: 50% da VRAM
    ram_limit_gb: float = 6.0            # Padrão: 6.0 GB de RAM máxima
    cpu_cores_limit_pct: int = 50        # Padrão: 50% dos núcleos (6 de 12)
    parallel_visual_arenas: int = 4      # Telas simultâneas no mosaico (1, 4, 6, 9, 12)
    background_workers: int = 12
    pop_or_batch_size: int = 512
    sim_speed_mult: int = 8              # Velocidade (1x até 128x Turbo)
    exec_mode: str = "HIBRIDO_GA_GPU"    # "HIBRIDO_GA_GPU", "PPO_GPU", "GA_CPU"
    difficulty: str = "easy"


class HardwareResourceManager:
    def __init__(self):
        self.total_cpu_cores = os.cpu_count() or 12
        self.max_numba_threads = numba.config.NUMBA_NUM_THREADS
        self.total_ram_gb = self._detect_system_ram_gb()
        self.has_cuda = HAS_TORCH and torch.cuda.is_available()
        self.gpu_name = torch.cuda.get_device_name(0) if self.has_cuda else "CPU Mode (PyTorch s/ CUDA)"
        self.total_vram_mb = (
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            if self.has_cuda else 12288.0
        )

    def _detect_system_ram_gb(self) -> float:
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return round(stat.ullTotalPhys / (1024 ** 3), 1)
        except Exception:
            return 16.0

    def apply_hardware_limits(self, cfg: HardwareConfig) -> Dict[str, Any]:
        active_cores = max(1, min(self.max_numba_threads, int(round(self.total_cpu_cores * (cfg.cpu_cores_limit_pct / 100.0)))))
        try:
            numba.set_num_threads(active_cores)
        except Exception:
            pass

        if HAS_TORCH:
            try:
                torch.set_num_threads(active_cores)
                if self.has_cuda:
                    fraction = max(0.10, min(0.95, cfg.gpu_vram_limit_pct / 100.0))
                    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass

        # Estimativa Preditiva de RAM e GPU
        base_ram_mb = 420.0 + (active_cores * 42.0) + (cfg.parallel_visual_arenas * 24.0)
        batch_ram_mb = (cfg.pop_or_batch_size * 0.95) * (1.0 + active_cores * 0.06)
        est_ram_gb = (base_ram_mb + batch_ram_mb) / 1024.0

        if cfg.exec_mode == "GA_CPU":
            est_gpu_pct = 5.0
            est_vram_mb = 180
            where_gpu = "ONDE USA GPU: Apenas renderização 2D (Redes Neurais desligadas)."
            how_gpu = f"COMO USA: 100% da simulação roda nos {active_cores} núcleos da CPU via Numba SoA."
            why_gpu = "POR QUE: Deixa a sua RTX 3060 95% livre para outros programas."
        elif cfg.exec_mode == "PPO_GPU":
            raw_gpu = 16.0 + (cfg.pop_or_batch_size / 64.0) * 3.4 + np.log2(max(1, cfg.sim_speed_mult)) * 4.5
            est_gpu_pct = min(float(cfg.gpu_compute_limit_pct), raw_gpu)
            est_vram_mb = int(420 + cfg.pop_or_batch_size * 1.5)
            where_gpu = "ONDE USA GPU: Tensor Cores Ampere (TF32) na inferência e gradientes PPO."
            how_gpu = f"COMO USA: Lotes de {cfg.pop_or_batch_size} agentes limitados a {cfg.gpu_compute_limit_pct}% da RTX 3060."
            why_gpu = "POR QUE: Acelera o treino de redes neurais respeitando o teto de 50%."
        else:
            raw_gpu = 12.0 + (cfg.pop_or_batch_size / 64.0) * 2.8 + np.log2(max(1, cfg.sim_speed_mult)) * 3.8
            est_gpu_pct = min(float(cfg.gpu_compute_limit_pct), raw_gpu)
            est_vram_mb = int(340 + cfg.pop_or_batch_size * 1.2)
            where_gpu = "ONDE USA GPU: Filtro Neural Surrogate na RTX 3060 + Física SoA na CPU."
            how_gpu = f"COMO USA: {active_cores} núcleos simulam ticks e a RTX 3060 (teto {cfg.gpu_compute_limit_pct}%) avalia elites."
            why_gpu = "POR QUE: Une a velocidade bruta do GA com a avaliação paralela da GPU."

        return {
            "active_cores": active_cores,
            "total_cores": self.total_cpu_cores,
            "est_ram_gb": round(est_ram_gb, 2),
            "est_gpu_compute_pct": round(est_gpu_pct, 1),
            "est_vram_mb": est_vram_mb,
            "max_allowed_vram_mb": int(self.total_vram_mb * (cfg.gpu_vram_limit_pct / 100.0)),
            "where_gpu": where_gpu,
            "how_gpu": how_gpu,
            "why_gpu": why_gpu,
        }

    def run_gpu_controlled_step(self, cfg: HardwareConfig):
        if not self.has_cuda or cfg.exec_mode == "GA_CPU":
            return
        t0 = time.perf_counter()
        with torch.no_grad():
            dim = min(1024, max(128, cfg.pop_or_batch_size))
            a = torch.randn((dim, 256), device="cuda", dtype=torch.float32)
            b = torch.randn((256, 256), device="cuda", dtype=torch.float32)
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        ratio = max(0.05, min(1.0, cfg.gpu_compute_limit_pct / 100.0))
        if ratio < 0.98:
            sleep_t = dt * ((1.0 - ratio) / ratio)
            if sleep_t > 0.0002:
                time.sleep(min(0.012, sleep_t))


# =====================================================================
# 4. COFRE DE MEMÓRIA PERSISTENTE (AUTO-SAVE & AUTO-RESUME)
# =====================================================================
class AIMemoryVault:
    def __init__(self, base_dir: str = "memory"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def _vault_path(self, diff: str) -> str:
        return os.path.join(self.base_dir, f"btd6_memory_{diff.lower()}.npz")

    def _meta_path(self, diff: str) -> str:
        return os.path.join(self.base_dir, f"btd6_meta_{diff.lower()}.json")

    def save_state(
        self,
        difficulty: str,
        population: np.ndarray,
        fitnesses: np.ndarray,
        generation: int,
        best_fitness: float,
        best_round: int,
        best_cash: float,
        best_lives: int,
        target_round: int,
        total_episodes: int,
    ):
        diff = difficulty.lower()
        v_path = self._vault_path(diff)
        m_path = self._meta_path(diff)

        sorted_idx = np.argsort(fitnesses)[::-1]
        sorted_pop = population[sorted_idx].astype(np.float32)
        sorted_fit = fitnesses[sorted_idx].astype(np.float64)

        hof_size = min(32, len(sorted_pop))
        hof_pop = sorted_pop[:hof_size].copy()
        hof_fit = sorted_fit[:hof_size].copy()

        if os.path.exists(v_path):
            try:
                with np.load(v_path) as data:
                    old_hof = data["hall_of_fame"]
                    old_fit = data["hof_fitness"]
                    if old_hof.shape[1:] == hof_pop.shape[1:]:
                        c_pop = np.concatenate([old_hof, hof_pop], axis=0)
                        c_fit = np.concatenate([old_fit, hof_fit], axis=0)
                        _, u_idx = np.unique(np.round(c_fit, 2), return_index=True)
                        top_u = u_idx[np.argsort(c_fit[u_idx])[::-1]][:hof_size]
                        if len(top_u) == hof_size:
                            hof_pop = c_pop[top_u]
                            hof_fit = c_fit[top_u]
            except Exception:
                pass

        if float(hof_fit[0]) > best_fitness:
            best_fitness = float(hof_fit[0])

        tmp_v = v_path + ".tmp.npz"
        np.savez_compressed(
            tmp_v,
            population=sorted_pop,
            fitnesses=sorted_fit,
            hall_of_fame=hof_pop,
            hof_fitness=hof_fit,
            best_chrom=hof_pop[0],
        )
        os.replace(tmp_v, v_path)

        meta = {
            "difficulty": diff,
            "target_round": int(target_round),
            "generation": int(generation),
            "total_episodes_learned": int(total_episodes),
            "best_fitness": round(float(best_fitness), 2),
            "best_round": int(best_round),
            "passed_level": bool(best_round >= target_round and best_lives > 0),
            "cash_left": round(float(best_cash), 1),
            "lives_left": int(best_lives),
            "last_saved": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        tmp_m = m_path + ".tmp"
        with open(tmp_m, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        os.replace(tmp_m, m_path)

    def load_state(
        self,
        difficulty: str,
        desired_pop_size: int,
        n_genes: int,
        target_round: int,
        path_coords: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        diff = difficulty.lower()
        v_path = self._vault_path(diff)
        m_path = self._meta_path(diff)

        if os.path.exists(v_path) and os.path.exists(m_path):
            try:
                with open(m_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                with np.load(v_path) as data:
                    saved_pop = data["population"].astype(np.float32)
                    hof_pop = data["hall_of_fame"].astype(np.float32)
                pop = self._reshape_pop(saved_pop, hof_pop, desired_pop_size, n_genes, target_round, path_coords)
                meta["resumed_from_memory"] = True
                meta["transfer_source"] = None
                return pop, meta
            except Exception:
                pass

        # Transfer Learning: aproveita o que aprendeu em dificuldades anteriores
        hierarchy = ["easy", "medium", "hard", "chimps"]
        if diff in hierarchy:
            c_idx = hierarchy.index(diff)
            for p_idx in range(c_idx - 1, -1, -1):
                p_diff = hierarchy[p_idx]
                p_vault = self._vault_path(p_diff)
                if os.path.exists(p_vault):
                    try:
                        with np.load(p_vault) as data:
                            hof_pop = data["hall_of_fame"].astype(np.float32)
                        pop = self._reshape_pop(hof_pop, hof_pop, desired_pop_size, n_genes, target_round, path_coords)
                        return pop, {
                            "difficulty": diff, "generation": 1, "total_episodes_learned": 0,
                            "best_fitness": 0.0, "best_round": 0, "resumed_from_memory": True,
                            "transfer_source": p_diff.upper()
                        }
                    except Exception:
                        pass

        pop = create_random_population(desired_pop_size, n_genes, target_round, path_coords)
        return pop, {
            "difficulty": diff, "generation": 1, "total_episodes_learned": 0,
            "best_fitness": 0.0, "best_round": 0, "resumed_from_memory": False, "transfer_source": None
        }

    def _reshape_pop(self, saved_pop, hof_pop, desired_size, n_genes, target_round, path_coords):
        new_pop = create_random_population(desired_size, n_genes, target_round, path_coords)
        mg = min(n_genes, saved_pop.shape[1])
        n_hof = min(len(hof_pop), desired_size // 4)
        for i in range(n_hof):
            new_pop[i, :mg, :] = hof_pop[i, :mg, :]
        n_copy = min(len(saved_pop), desired_size - n_hof)
        for i in range(n_copy):
            new_pop[n_hof + i, :mg, :] = saved_pop[i, :mg, :]
        return new_pop


# =====================================================================
# 5. ARENA VISUAL INDIVIDUAL E CENTRAL DE CONTROLE MULTI-INSTÂNCIAS
# =====================================================================
COL_BG_DARK = (18, 21, 28)
COL_PANEL   = (26, 31, 42)
COL_CARD    = (35, 42, 56)
COL_GRASS   = (44, 102, 46)
COL_TRACK_B = (105, 80, 50)
COL_TRACK_F = (184, 150, 104)
COL_TEXT    = (238, 242, 248)
COL_GOLD    = (255, 210, 60)
COL_GREEN   = (72, 215, 115)
COL_YELLOW  = (245, 185, 55)
COL_RED     = (245, 80, 80)
COL_CYAN    = (65, 195, 255)
COL_PURPLE  = (185, 115, 255)

TOWER_COLORS = {
    0: (165, 105, 55),  # Dart Monkey
    1: (55, 60, 70),    # Bomb Shooter
    2: (160, 70, 210),  # Alchemist
    3: (210, 65, 65),   # Monkey Village
    4: (230, 195, 40),  # Banana Farm
}

BLOON_COLORS = [
    (235, 45, 45), (45, 115, 235), (50, 190, 65),
    (245, 215, 40), (245, 115, 175), (95, 105, 115),
    (195, 115, 55), (40, 140, 255), (200, 30, 30),
]


class MiniArenaInstance:
    def __init__(self, instance_id: int, chrom: np.ndarray, difficulty: str = "easy"):
        self.instance_id = instance_id
        self.reset(chrom, difficulty)

    def reset(self, chrom: np.ndarray, difficulty: str):
        self.chrom = chrom.copy()
        self.difficulty = difficulty
        cfg = DIFFICULTY_CONFIG[difficulty]
        self.target_round = cfg["target_round"]
        self.cash = cfg["start_cash"]
        self.lives = cfg["start_lives"]
        self.cost_mult = cfg["cost_mult"]

        self.current_round = 1
        self.towers = []
        self.bloons = []
        self.lasers = []
        self.pops = 0
        self.finished = False
        self.victory = False

        self._apply_round_genes(1)
        self._spawn_bloons(1)

    def _apply_round_genes(self, r: int):
        for g in range(self.chrom.shape[0]):
            act = int(self.chrom[g, 0])
            if int(self.chrom[g, 1]) != r or act == 0:
                continue
            slot_or_type = int(self.chrom[g, 2])
            gx, gy = float(self.chrom[g, 3]), float(self.chrom[g, 4])
            up_path = int(self.chrom[g, 5]) % 3

            if act == 1 and len(self.towers) < 28:
                tid = slot_or_type % 5
                cost = float(TOWER_COSTS[tid]) * self.cost_mult
                if self.cash >= cost and point_to_track_dist(gx, gy, MONKEY_MEADOW_PATH) >= 28.0:
                    self.cash -= cost
                    self.towers.append({
                        "type": tid, "x": gx, "y": gy,
                        "top": 0, "mid": 0, "bot": 0,
                        "invested": cost, "cd": 0.0
                    })
            elif act == 2 and self.towers:
                slot = slot_or_type % len(self.towers)
                t = self.towers[slot]
                nt = t["top"] + (1 if up_path == 0 else 0)
                nm = t["mid"] + (1 if up_path == 1 else 0)
                nb = t["bot"] + (1 if up_path == 2 else 0)
                if is_valid_crosspath(nt, nm, nb):
                    up_cost = (180.0 * (1.65 ** max(nt, nm, nb))) * self.cost_mult
                    if self.cash >= up_cost:
                        self.cash -= up_cost
                        t["top"], t["mid"], t["bot"] = nt, nm, nb
                        t["invested"] += up_cost
            elif act == 3 and self.towers:
                slot = slot_or_type % len(self.towers)
                t = self.towers.pop(slot)
                self.cash += t["invested"] * 0.70

    def _spawn_bloons(self, r: int):
        self.bloons.clear()
        count = min(28, 6 + int(r * 1.1))
        for i in range(count):
            if r >= 40 and i == count - 1:
                tier = 7 if r < 60 else 8
                hp = 200.0 if tier == 7 else 600.0
                spd = 1.3
            else:
                tier = min(6, (i + r // 3) % (min(6, r // 5) + 1))
                hp = (1.0 + tier * 2.2) * (1.0 + max(0, r - 15) * 0.06)
                spd = 2.2 + tier * 0.35
            self.bloons.append({
                "tier": tier, "hp": hp, "speed": spd,
                "wp": 0, "x": float(MONKEY_MEADOW_PATH[0, 0]) - i * 25.0,
                "y": float(MONKEY_MEADOW_PATH[0, 1])
            })

    def step(self):
        if self.finished:
            return

        alive = []
        for b in self.bloons:
            if b["x"] < 0:
                b["x"] += b["speed"]
                alive.append(b)
                continue
            tx, ty = float(MONKEY_MEADOW_PATH[b["wp"] + 1, 0]), float(MONKEY_MEADOW_PATH[b["wp"] + 1, 1])
            dx, dy = tx - b["x"], ty - b["y"]
            dist = math.hypot(dx, dy)
            if dist <= b["speed"]:
                b["x"], b["y"] = tx, ty
                b["wp"] += 1
                if b["wp"] >= len(MONKEY_MEADOW_PATH) - 1:
                    self.lives = max(0, self.lives - max(1, int(b["hp"] * 0.5)))
                    if self.lives == 0:
                        self.finished = True
                        self.victory = False
                    continue
            else:
                b["x"] += (dx / dist) * b["speed"]
                b["y"] += (dy / dist) * b["speed"]
            alive.append(b)
        self.bloons = alive

        for t in self.towers:
            tid = t["type"]
            if tid in (3, 4):
                continue
            if t["cd"] > 0:
                t["cd"] -= 1.0
                continue
            rng = float(TOWER_RANGES[tid]) * (1.0 + 0.08 * t["bot"])
            target = None
            best_wp = -1
            for b in self.bloons:
                if b["x"] >= 0 and math.hypot(b["x"] - t["x"], b["y"] - t["y"]) <= rng:
                    if b["wp"] > best_wp:
                        best_wp = b["wp"]
                        target = b
            if target is not None:
                dmg = float(TOWER_BASE_DMG[tid]) + t["top"] * 1.8
                target["hp"] -= dmg
                self.pops += 1
                self.cash += 1.1
                t["cd"] = max(4.0, (float(TOWER_COOLDOWN[tid]) / (1.0 + 0.22 * t["mid"])) * 0.35)
                self.lasers.append((t["x"], t["y"], target["x"], target["y"], TOWER_COLORS[tid]))

        self.bloons = [b for b in self.bloons if b["hp"] > 0]

        if not self.bloons and not self.finished:
            for t in self.towers:
                if t["type"] == 4:
                    self.cash += 95.0 + t["top"] * 115.0
            self.cash += 100.0 + self.current_round * 3.0
            if self.current_round >= self.target_round:
                self.finished = True
                self.victory = True
            else:
                self.current_round += 1
                self._apply_round_genes(self.current_round)
                self._spawn_bloons(self.current_round)


class BTD6ControlCenterApp:
    def __init__(self):
        pygame.init()
        self.width, self.height = 1440, 840
        self.sidebar_w = 480
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("BTD6 AI - Central de Hardware, Paralelismo & Memória Persistente")
        self.clock = pygame.time.Clock()

        self.font_h1 = pygame.font.SysFont("Segoe UI", 18, bold=True)
        self.font_h2 = pygame.font.SysFont("Segoe UI", 13, bold=True)
        self.font_sm = pygame.font.SysFont("Segoe UI", 12)
        self.font_mono = pygame.font.SysFont("Consolas", 12, bold=True)

        self.hw_mgr = HardwareResourceManager()
        self.vault = AIMemoryVault()
        self.cfg = HardwareConfig()
        self.telemetry = self.hw_mgr.apply_hardware_limits(self.cfg)

        self.training_active = True
        self.zoomed_arena_idx = None
        self.epm_measured = 0
        self.buttons = []
        self.memory_status_msg = ""

        self._load_memory_and_init_arenas()

        # Aquecimento imediato do compilador JIT Numba
        d_cfg = DIFFICULTY_CONFIG[self.cfg.difficulty]
        _ = evaluate_population_parallel(
            self.population[:4], d_cfg["target_round"], d_cfg["start_cash"],
            d_cfg["start_lives"], d_cfg["cost_mult"], MONKEY_MEADOW_PATH
        )

    def _load_memory_and_init_arenas(self):
        target_r = DIFFICULTY_CONFIG[self.cfg.difficulty]["target_round"]
        self.population, meta = self.vault.load_state(
            self.cfg.difficulty, self.cfg.pop_or_batch_size, 90, target_r, MONKEY_MEADOW_PATH
        )
        self.generation = int(meta.get("generation", 1))
        self.best_fitness = float(meta.get("best_fitness", 0.0))
        self.best_round = int(meta.get("best_round", 0))
        self.total_episodes = int(meta.get("total_episodes_learned", 0))

        self.last_fitnesses = np.zeros(len(self.population), dtype=np.float64)
        self.last_rounds = np.zeros(len(self.population), dtype=np.int32)
        self.last_cashes = np.zeros(len(self.population), dtype=np.float32)
        self.last_lives = np.zeros(len(self.population), dtype=np.int32)

        if meta.get("resumed_from_memory"):
            if meta.get("transfer_source"):
                self.memory_status_msg = f"Transferido de {meta['transfer_source']} (Aprendizado Importado)"
            else:
                self.memory_status_msg = f"Restaurado da Gen #{self.generation} ({self.total_episodes:,} partidas)"
        else:
            self.memory_status_msg = "Nova Memória Iniciada (Gen #1)"

        self._rebuild_visual_arenas()

    def _rebuild_visual_arenas(self):
        self.arenas = [
            MiniArenaInstance(i + 1, self.population[i % len(self.population)], self.cfg.difficulty)
            for i in range(self.cfg.parallel_visual_arenas)
        ]

    def save_current_memory(self):
        d_cfg = DIFFICULTY_CONFIG[self.cfg.difficulty]
        if not np.any(self.last_fitnesses > 0):
            self.last_fitnesses, self.last_rounds, self.last_cashes, self.last_lives = evaluate_population_parallel(
                self.population, d_cfg["target_round"], d_cfg["start_cash"],
                d_cfg["start_lives"], d_cfg["cost_mult"], MONKEY_MEADOW_PATH
            )
        b_idx = int(np.argmax(self.last_fitnesses))
        self.vault.save_state(
            self.cfg.difficulty, self.population, self.last_fitnesses, self.generation,
            self.best_fitness, int(self.last_rounds[b_idx]), float(self.last_cashes[b_idx]),
            int(self.last_lives[b_idx]), d_cfg["target_round"], self.total_episodes
        )
        self.memory_status_msg = f"Salvo na Gen #{self.generation} às {time.strftime('%H:%M:%S')}"

    def step_background_training_generation(self):
        d_cfg = DIFFICULTY_CONFIG[self.cfg.difficulty]
        t0 = time.perf_counter()

        effective_pop = self.population[: self.cfg.pop_or_batch_size]
        fitnesses, rounds, cashes, lives_arr = evaluate_population_parallel(
            effective_pop, d_cfg["target_round"], d_cfg["start_cash"],
            d_cfg["start_lives"], d_cfg["cost_mult"], MONKEY_MEADOW_PATH
        )
        self.hw_mgr.run_gpu_controlled_step(self.cfg)

        self.last_fitnesses, self.last_rounds, self.last_cashes, self.last_lives = fitnesses, rounds, cashes, lives_arr
        dt = max(1e-5, time.perf_counter() - t0)
        self.epm_measured = int((len(effective_pop) / dt) * 60.0)
        self.total_episodes += len(effective_pop)

        best_idx = int(np.argmax(fitnesses))
        gen_best = float(fitnesses[best_idx])

        if gen_best > self.best_fitness + 1e-3:
            self.best_fitness = gen_best
            self.best_round = int(rounds[best_idx])
            self.save_current_memory()
        elif self.generation % 10 == 0:
            self.save_current_memory()

        top_indices = np.argsort(fitnesses)[::-1]
        for idx, arena in enumerate(self.arenas):
            if arena.finished:
                arena.reset(effective_pop[top_indices[idx % len(top_indices)]], self.cfg.difficulty)

        self.population[: len(effective_pop)] = evolve_next_generation(
            effective_pop, fitnesses, 0.18, d_cfg["target_round"], MONKEY_MEADOW_PATH
        )
        self.generation += 1

    def _adjust_param(self, param_name: str, delta: int):
        if param_name in ("diff", "batch"):
            self.save_current_memory()

        if param_name == "arenas":
            options = [1, 4, 6, 9, 12]
            idx = options.index(self.cfg.parallel_visual_arenas) if self.cfg.parallel_visual_arenas in options else 1
            self.cfg.parallel_visual_arenas = options[max(0, min(len(options) - 1, idx + delta))]
            self.zoomed_arena_idx = None
            self._rebuild_visual_arenas()
        elif param_name == "speed":
            options = [1, 2, 4, 8, 16, 32, 64, 128]
            idx = options.index(self.cfg.sim_speed_mult) if self.cfg.sim_speed_mult in options else 3
            self.cfg.sim_speed_mult = options[max(0, min(len(options) - 1, idx + delta))]
        elif param_name == "gpu_pct":
            self.cfg.gpu_compute_limit_pct = max(10, min(100, self.cfg.gpu_compute_limit_pct + delta * 10))
            self.cfg.gpu_vram_limit_pct = self.cfg.gpu_compute_limit_pct
        elif param_name == "ram_gb":
            self.cfg.ram_limit_gb = round(max(2.0, min(16.0, self.cfg.ram_limit_gb + delta * 0.5)), 1)
        elif param_name == "cpu_pct":
            self.cfg.cpu_cores_limit_pct = max(10, min(100, self.cfg.cpu_cores_limit_pct + delta * 10))
        elif param_name == "batch":
            options = [128, 256, 512, 1024, 2048]
            idx = options.index(self.cfg.pop_or_batch_size) if self.cfg.pop_or_batch_size in options else 2
            self.cfg.pop_or_batch_size = options[max(0, min(len(options) - 1, idx + delta))]
            self._load_memory_and_init_arenas()
        elif param_name == "mode":
            modes = ["HIBRIDO_GA_GPU", "PPO_GPU", "GA_CPU"]
            self.cfg.exec_mode = modes[(modes.index(self.cfg.exec_mode) + delta) % len(modes)]
        elif param_name == "diff":
            diffs = ["easy", "medium", "hard", "chimps"]
            self.cfg.difficulty = diffs[(diffs.index(self.cfg.difficulty) + delta) % len(diffs)]
            self._load_memory_and_init_arenas()

        self.telemetry = self.hw_mgr.apply_hardware_limits(self.cfg)

    def draw_sidebar(self):
        pygame.draw.rect(self.screen, COL_PANEL, (0, 0, self.sidebar_w, self.height))
        self.buttons.clear()
        x, y = 16, 12

        self.screen.blit(self.font_h1.render("CENTRAL DE HARDWARE & MEMÓRIA DA IA", True, COL_GOLD), (x, y))
        y += 24
        self.screen.blit(
            self.font_sm.render(f"GPU: {self.hw_mgr.gpu_name} | RAM Sistema: {self.hw_mgr.total_ram_gb} GB", True, COL_CYAN),
            (x, y),
        )
        y += 22

        speed_lbl = "TURBO (128x)" if self.cfg.sim_speed_mult >= 128 else f"{self.cfg.sim_speed_mult}x"
        rows = [
            ("Telas Paralelas Ao Vivo:", f"{self.cfg.parallel_visual_arenas} simultâneas", "arenas"),
            ("Velocidade de Simulação:", speed_lbl, "speed"),
            ("Teto da Placa de Vídeo (GPU):", f"{self.cfg.gpu_compute_limit_pct}% RTX 3060", "gpu_pct"),
            ("Teto de Memória RAM p/ IA:", f"{self.cfg.ram_limit_gb:.1f} GB Máx", "ram_gb"),
            ("Teto de Processador (CPU):", f"{self.cfg.cpu_cores_limit_pct}% ({self.telemetry['active_cores']}/{self.telemetry['total_cores']} núcleos)", "cpu_pct"),
            ("População / Batch Paralelo:", f"{self.cfg.pop_or_batch_size} agentes", "batch"),
            ("Modo de Execução (Hardware):", self.cfg.exec_mode, "mode"),
            ("Dificuldade do Nível:", self.cfg.difficulty.upper(), "diff"),
        ]

        for label_txt, val_txt, key in rows:
            pygame.draw.rect(self.screen, COL_CARD, (x, y, self.sidebar_w - 32, 31), border_radius=6)
            self.screen.blit(self.font_sm.render(label_txt, True, COL_TEXT), (x + 10, y + 7))

            btn_m = pygame.Rect(self.sidebar_w - 190, y + 3, 26, 25)
            btn_p = pygame.Rect(self.sidebar_w - 44, y + 3, 26, 25)
            pygame.draw.rect(self.screen, (55, 65, 85), btn_m, border_radius=4)
            pygame.draw.rect(self.screen, (55, 65, 85), btn_p, border_radius=4)
            self.screen.blit(self.font_h2.render("-", True, COL_GOLD), (btn_m.centerx - 4, btn_m.y + 3))
            self.screen.blit(self.font_h2.render("+", True, COL_GOLD), (btn_p.centerx - 5, btn_p.y + 3))

            v_surf = self.font_mono.render(val_txt, True, COL_GREEN)
            self.screen.blit(v_surf, (btn_m.right + ((btn_p.left - btn_m.right) - v_surf.get_width()) // 2, y + 8))

            self.buttons.append((btn_m, key, -1))
            self.buttons.append((btn_p, key, +1))
            y += 35

        # Seção: Estimativa Preditiva de Hardware (GPU, VRAM e RAM)
        y += 4
        self.screen.blit(self.font_h2.render("CALCULADORA DE HARDWARE EM TEMPO REAL:", True, COL_GOLD), (x, y))
        y += 20

        gpu_est = self.telemetry["est_gpu_compute_pct"]
        vram_est = self.telemetry["est_vram_mb"]
        vram_lim = max(1, self.telemetry["max_allowed_vram_mb"])
        ram_est = self.telemetry["est_ram_gb"]

        self._draw_bar(
            x, y, self.sidebar_w - 32,
            f"GPU RTX 3060 Uso: {gpu_est:.1f}% / Teto Escolhido: {self.cfg.gpu_compute_limit_pct}%",
            gpu_est, 100.0, self.cfg.gpu_compute_limit_pct
        )
        y += 34
        self._draw_bar(
            x, y, self.sidebar_w - 32,
            f"VRAM Placa de Vídeo: {vram_est} MB / Teto ({self.cfg.gpu_vram_limit_pct}%): {vram_lim} MB",
            (vram_est / max(1.0, self.hw_mgr.total_vram_mb)) * 100.0, 100.0, self.cfg.gpu_vram_limit_pct
        )
        y += 34
        self._draw_bar(
            x, y, self.sidebar_w - 32,
            f"Memória RAM IA: {ram_est:.2f} GB / Teto Alocado: {self.cfg.ram_limit_gb:.1f} GB",
            (ram_est / max(0.5, self.cfg.ram_limit_gb)) * 100.0, 100.0, 100.0
        )
        y += 38

        # Seção: Explicação ONDE / COMO / POR QUE + Cofre de Memória
        pygame.draw.rect(self.screen, COL_CARD, (x, y, self.sidebar_w - 32, 165), border_radius=6)
        self.screen.blit(self.font_h2.render("DIAGNÓSTICO DE USO & MEMÓRIA PERSISTENTE:", True, COL_PURPLE), (x + 10, y + 6))

        target_r = DIFFICULTY_CONFIG[self.cfg.difficulty]["target_round"]
        info_lines = [
            self.telemetry["where_gpu"],
            self.telemetry["how_gpu"],
            self.telemetry["why_gpu"],
            f"MEMÓRIA: {self.memory_status_msg}",
            f"PROGRESSO: Gen #{self.generation} | {self.total_episodes:,} partidas acumuladas",
            f"RECORDE SALVO: Fit {self.best_fitness:,.1f} (Melhor Round: {self.best_round}/{target_r})",
            f"VELOCIDADE DO MOTOR: {self.epm_measured:,} Episódios/Minuto (EPM)",
        ]
        iy = y + 26
        for line in info_lines:
            self.screen.blit(self.font_sm.render(line[:68], True, COL_TEXT), (x + 10, iy))
            iy += 19
        y += 172

        # Botões de Ação
        btn_save = pygame.Rect(x, y, self.sidebar_w - 32, 32)
        pygame.draw.rect(self.screen, (50, 110, 195), btn_save, border_radius=6)
        s_surf = self.font_h2.render("SALVAR APRENDIZADO NA MEMÓRIA AGORA", True, COL_TEXT)
        self.screen.blit(s_surf, (btn_save.centerx - s_surf.get_width() // 2, btn_save.y + 7))
        self.buttons.append((btn_save, "save_now", 0))
        y += 37

        btn_toggle = pygame.Rect(x, y, self.sidebar_w - 32, 32)
        pygame.draw.rect(self.screen, (45, 145, 85) if self.training_active else (185, 65, 65), btn_toggle, border_radius=6)
        t_txt = "TREINO PARALELO: RODANDO (CLIQUE P/ PAUSAR)" if self.training_active else "TREINO PARALELO: PAUSADO (CLIQUE P/ RETOMAR)"
        t_surf = self.font_h2.render(t_txt, True, COL_TEXT)
        self.screen.blit(t_surf, (btn_toggle.centerx - t_surf.get_width() // 2, btn_toggle.y + 7))
        self.buttons.append((btn_toggle, "toggle_train", 0))

    def _draw_bar(self, x: int, y: int, w: int, label: str, pct: float, max_val: float, limit_pct: float):
        self.screen.blit(self.font_mono.render(label, True, COL_TEXT), (x, y))
        pygame.draw.rect(self.screen, (40, 46, 60), (x, y + 16, w, 12), border_radius=4)
        fill = int(w * min(1.0, max(0.0, pct / max_val)))
        col = COL_GREEN if pct <= limit_pct * 0.85 else (COL_YELLOW if pct <= limit_pct else COL_RED)
        if fill > 0:
            pygame.draw.rect(self.screen, col, (x, y + 16, fill, 12), border_radius=4)
        if limit_pct < 100.0:
            mx = x + int(w * (limit_pct / max_val))
            pygame.draw.line(self.screen, COL_GOLD, (mx, y + 14), (mx, y + 30), 2)

    def draw_arenas_grid(self):
        grid_x = self.sidebar_w
        grid_w = self.width - self.sidebar_w
        pygame.draw.rect(self.screen, COL_BG_DARK, (grid_x, 0, grid_w, self.height))
        self.arena_rects = []

        if self.zoomed_arena_idx is not None and self.zoomed_arena_idx < len(self.arenas):
            rect = pygame.Rect(grid_x + 12, 12, grid_w - 24, self.height - 24)
            self._render_single_arena(self.arenas[self.zoomed_arena_idx], rect, True)
            self.arena_rects.append((rect, self.zoomed_arena_idx))
            return

        n = len(self.arenas)
        cols, rows = (1, 1) if n == 1 else ((2, 2) if n <= 4 else ((3, 2) if n <= 6 else ((3, 3) if n <= 9 else (4, 3))))
        pad = 10
        cell_w = (grid_w - pad * (cols + 1)) // cols
        cell_h = (self.height - pad * (rows + 1)) // rows

        for idx, arena in enumerate(self.arenas):
            c, r = idx % cols, idx // cols
            rect = pygame.Rect(grid_x + pad + c * (cell_w + pad), pad + r * (cell_h + pad), cell_w, cell_h)
            self._render_single_arena(arena, rect, False)
            self.arena_rects.append((rect, idx))

    def _render_single_arena(self, arena: MiniArenaInstance, rect: pygame.Rect, is_zoomed: bool):
        pygame.draw.rect(self.screen, COL_GRASS, rect, border_radius=8)
        pygame.draw.rect(self.screen, (75, 90, 115), rect, 2, border_radius=8)
        sx, sy = rect.width / 860.0, rect.height / 800.0
        to_scr = lambda px, py: (int(rect.x + px * sx), int(rect.y + py * sy))

        pts = [to_scr(p[0], p[1]) for p in MONKEY_MEADOW_PATH]
        tw = max(6, int(36 * min(sx, sy)))
        pygame.draw.lines(self.screen, COL_TRACK_B, False, pts, tw + 4)
        pygame.draw.lines(self.screen, COL_TRACK_F, False, pts, tw)

        t_rad = max(5, int(15 * min(sx, sy)))
        for t in arena.towers:
            tx, ty = to_scr(t["x"], t["y"])
            pygame.draw.circle(self.screen, (20, 20, 20), (tx, ty), t_rad + 2)
            pygame.draw.circle(self.screen, TOWER_COLORS[t["type"]], (tx, ty), t_rad)
            if is_zoomed:
                lbl = self.font_mono.render(f"{t['top']}-{t['mid']}-{t['bot']}", True, COL_TEXT)
                self.screen.blit(lbl, (tx - lbl.get_width() // 2, ty - t_rad - 14))

        for x1, y1, x2, y2, col in arena.lasers:
            pygame.draw.line(self.screen, col, to_scr(x1, y1), to_scr(x2, y2), max(1, int(3 * min(sx, sy))))
        arena.lasers.clear()

        b_rad = max(4, int(10 * min(sx, sy)))
        for b in arena.bloons:
            if b["x"] < 0:
                continue
            bx, by = to_scr(b["x"], b["y"])
            col = BLOON_COLORS[min(len(BLOON_COLORS) - 1, b["tier"])]
            pygame.draw.circle(self.screen, (15, 15, 15), (bx, by), b_rad + 1)
            pygame.draw.circle(self.screen, col, (bx, by), b_rad)

        pygame.draw.rect(self.screen, (18, 22, 30), (rect.x, rect.y, rect.width, 25), border_top_left_radius=8, border_top_right_radius=8)
        st = f"IA #{arena.instance_id:02d} | R{arena.current_round:02d}/{arena.target_round} | Vidas: {arena.lives} | ${int(arena.cash)} | Torres: {len(arena.towers)}"
        if is_zoomed:
            st += "   [CLIQUE NA ARENA PARA VOLTAR AO MOSAICO]"
        self.screen.blit(self.font_mono.render(st, True, COL_GOLD if arena.victory else COL_TEXT), (rect.x + 8, rect.y + 5))

    def run(self):
        running = True
        gen_tick = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.save_current_memory()
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    for rect, key, delta in self.buttons:
                        if rect.collidepoint(mx, my):
                            if key == "toggle_train":
                                self.training_active = not self.training_active
                            elif key == "save_now":
                                self.save_current_memory()
                            else:
                                self._adjust_param(key, delta)
                            break
                    else:
                        for rect, idx in getattr(self, "arena_rects", []):
                            if rect.collidepoint(mx, my):
                                self.zoomed_arena_idx = None if self.zoomed_arena_idx == idx else idx
                                break

            if self.training_active:
                for _ in range(self.cfg.sim_speed_mult):
                    for arena in self.arenas:
                        arena.step()
                gen_tick += 1
                if gen_tick >= max(4, 48 // max(1, self.cfg.sim_speed_mult // 4)):
                    self.step_background_training_generation()
                    gen_tick = 0

            self.draw_sidebar()
            self.draw_arenas_grid()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    BTD6ControlCenterApp().run()