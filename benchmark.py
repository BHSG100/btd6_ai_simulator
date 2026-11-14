from __future__ import annotations

import os
from pathlib import Path
import time
import numpy as np

from ai.ga_evaluator import GeneOp, ParallelGAEvaluator
from ai.gym_env import BTD6GymEnv
from core.arenas import TargetingMode
from core.engine import GameEngine
from utils.data_loader import GameDataLoader


def run_verification_tests(
    grid_w: int,
    grid_h: int,
    waypoints: np.ndarray,
    path_cells: np.ndarray,
    towers_db: dict,
    bloon_lookup,
    rounds_schedule: list,
) -> None:
    """Valida trava de crosspathing 5-2-0, venda/reembolso e Action Masking do Gym."""
    eng = GameEngine(
        grid_w, grid_h, waypoints, path_cells, towers_db, bloon_lookup, rounds_schedule, starting_cash=20000
    )

    # 1. Teste de Crosspathing 5-2-0
    slot = eng.action_place_tower("dart_monkey", 5, 8)
    assert slot == 0
    assert eng.action_upgrade_tower(slot, 0)  # 1-0-0
    assert eng.action_upgrade_tower(slot, 0)  # 2-0-0
    assert eng.action_upgrade_tower(slot, 1)  # 2-1-0
    assert eng.action_upgrade_tower(slot, 1)  # 2-2-0
    # Tentar abrir o 3º caminho (2-2-1) deve falhar
    assert not eng.action_upgrade_tower(slot, 2)
    # Evoluir caminho principal para 3-2-0 deve funcionar
    assert eng.action_upgrade_tower(slot, 0)  # 3-2-0
    # Tentar evoluir o secondary crosspath para 3-3-0 deve falhar
    assert not eng.action_upgrade_tower(slot, 1)

    # 2. Teste de Venda, Reembolso (70%) e Liberação de Grid + Aura Village
    cash_before_sell = eng.cash
    invested = int(eng.towers.total_invested[slot])
    refund = eng.action_sell_tower(slot)
    assert refund == int(invested * 0.70)
    assert eng.cash == cash_before_sell + refund
    assert eng.grid.is_cell_free(5, 8)

    # 3. Teste de consistência do BTD6GymEnv e Action Masking
    env = BTD6GymEnv(grid_w, grid_h, waypoints, path_cells, towers_db, bloon_lookup, rounds_schedule)
    obs, _ = env.reset(seed=123)
    rng = np.random.default_rng(123)
    for _ in range(150):
        masks = env.action_masks()
        valid_actions = np.flatnonzero(masks)
        assert valid_actions.size > 0
        # Prioriza testar ações de compra/upgrade/venda além da ação 0
        non_zero = valid_actions[valid_actions > 0]
        act = int(rng.choice(non_zero)) if non_zero.size > 0 and rng.random() < 0.7 else 0
        obs, reward, terminated, truncated, info = env.step(act)
        if terminated:
            obs, _ = env.reset()


def main() -> None:
    print("=" * 72)
    print(" BTD6 HIGH-PERFORMANCE AI ENGINE v2 - UPGRADES 5-2-0, VENDA E BENCHMARK")
    print("=" * 72)

    root = Path(__file__).resolve().parent
    towers_db = GameDataLoader.load_towers(root / "data" / "towers.json")
    bloons_db, bloon_lookup = GameDataLoader.load_bloons(root / "data" / "bloons.json")
    rounds_schedule = GameDataLoader.load_and_compile_rounds(
        root / "data" / "rounds.json", bloons_db
    )

    # Utiliza diretamente as propriedades do Monkey Meadow
    grid_w, grid_h = 32, 32
    waypoints = np.array(
        [[0.0, 5.0], [8.0, 5.0], [12.0, 10.0], [12.0, 22.0],
         [8.0, 26.0], [16.0, 26.0], [22.0, 20.0], [22.0, 10.0],
         [28.0, 5.0], [31.0, 5.0]],
        dtype=np.float32,
    )
    path_cells = np.array(
        [[0, 5], [1, 5], [2, 5], [3, 5], [4, 5], [5, 5], [6, 5], [7, 5], [8, 5],
         [9, 6], [10, 7], [11, 8], [12, 10], [12, 12], [12, 14], [12, 16], [12, 18], [12, 20], [12, 22],
         [11, 24], [10, 25], [8, 26], [9, 26], [10, 26], [12, 26], [14, 26], [16, 26],
         [18, 25], [20, 23], [22, 20], [22, 18], [22, 16], [22, 14], [22, 12], [22, 10],
         [23, 8], [25, 6], [28, 5], [29, 5], [30, 5], [31, 5]], dtype=np.int32
    )

    print("[1/3] Validando trava 5-2-0, venda/reembolso e Action Masking Gym...")
    run_verification_tests(grid_w, grid_h, waypoints, path_cells, towers_db, bloon_lookup, rounds_schedule)
    print("      Todas as asserções de regras e máscaras passaram com sucesso!")

    engine = GameEngine(
        grid_width=grid_w,
        grid_height=grid_h,
        waypoints=waypoints,
        path_cells=path_cells,
        towers_catalog=towers_db,
        bloon_lookup=bloon_lookup,
        rounds_schedule=rounds_schedule,
        starting_cash=10000,
    )

    # Warm-up LLVM
    engine.action_place_tower("monkey_village", 10, 8, top_tier=2, mid_tier=2)
    engine.action_place_tower("alchemist", 12, 8, top_tier=2)
    engine.action_place_tower("bomb_shooter", 14, 7, top_tier=2)
    engine.action_place_tower("dart_monkey", 16, 7, top_tier=3, mid_tier=2, target_mode=TargetingMode.STRONG)
    while not engine.is_game_over and not engine.is_victory:
        engine.run_round(dt=0.05)

    episodes = 200
    print(f"[2/3] Benchmark Single-Core ({episodes} partidas completas)...")
    t_single = time.perf_counter()
    total_ticks = 0
    for _ in range(episodes):
        engine.reset()
        engine.action_place_tower("monkey_village", 10, 8, top_tier=2, mid_tier=2)
        engine.action_place_tower("alchemist", 12, 8, top_tier=2)
        engine.action_place_tower("bomb_shooter", 14, 7, top_tier=2)
        engine.action_place_tower("dart_monkey", 16, 7, top_tier=3, mid_tier=2)
        while not engine.is_game_over and not engine.is_victory:
            engine.run_round(dt=0.05)
        total_ticks += engine.ticks_elapsed

    elapsed_single = time.perf_counter() - t_single
    print(f"      -> Vazão Single-Core : {total_ticks / elapsed_single:>12,.0f} TPS")
    print(f"      -> Episódios/Minuto  : {(episodes / elapsed_single) * 60:>12,.0f} EPM")

    # Benchmark Multi-Core com genes polimórficos (PLACE, UPGRADE, SELL)
    cpu_cores = os.cpu_count() or 4
    pop_size = 500
    rng = np.random.default_rng(42)
    num_t = len(towers_db)
    population = []
    for _ in range(pop_size):
        g_place = np.column_stack((
            rng.integers(0, 3, size=4),
            np.full(4, GeneOp.PLACE),
            rng.integers(0, num_t, size=4),
            rng.integers(2, 28, size=4),
            rng.integers(2, 28, size=4),
            rng.integers(0, 4, size=4),
        ))
        g_upg = np.column_stack((
            rng.integers(1, len(rounds_schedule), size=4),
            np.full(4, GeneOp.UPGRADE),
            rng.integers(0, 4, size=4),
            rng.integers(0, 3, size=4),
            rng.integers(-1, 4, size=4),
            np.zeros(4, dtype=np.int32),
        ))
        population.append(np.vstack((g_place, g_upg)).astype(np.int32))

    print(f"[3/3] Benchmark Multi-Core GA Polimórfico ({pop_size} genomas em {cpu_cores} núcleos)...")
    with ParallelGAEvaluator(
        grid_w, grid_h, waypoints, path_cells, towers_db, bloon_lookup, rounds_schedule, max_workers=cpu_cores
    ) as evaluator:
        t_mp = time.perf_counter()
        fitness_scores = evaluator.evaluate_population(population, chunksize=20)
        elapsed_mp = time.perf_counter() - t_mp

    print(f"      -> Episódios/Minuto  : {(pop_size / elapsed_mp) * 60:>12,.0f} EPM")
    print(f"      -> Melhor Fitness    : {float(np.max(fitness_scores)):>12,.1f}")
    print("=" * 72)


if __name__ == "__main__":
    main()