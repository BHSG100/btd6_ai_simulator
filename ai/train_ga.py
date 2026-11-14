import os
import sys
import time
import signal
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.simulator import (
    MONKEY_MEADOW_PATH,
    TOWER_COSTS,
    TOWER_RANGES,
    TOWER_BASE_DMG,
    TOWER_BASE_PIERCE,
    TOWER_COOLDOWN,
    is_valid_crosspath,
    point_to_track_dist,
    evaluate_match,
    evaluate_population_parallel,
)
from utils.memory_vault import AIMemoryVault

DIFFICULTY_CONFIG = {
    "easy":   {"target_round": 40,  "start_cash": 650.0, "start_lives": 200, "cost_mult": 0.85},
    "medium": {"target_round": 60,  "start_cash": 650.0, "start_lives": 150, "cost_mult": 1.00},
    "hard":   {"target_round": 80,  "start_cash": 650.0, "start_lives": 100, "cost_mult": 1.08},
    "chimps": {"target_round": 100, "start_cash": 650.0, "start_lives": 1,   "cost_mult": 1.08},
}

ACT_NONE, ACT_PLACE, ACT_UPGRADE, ACT_SELL = 0, 1, 2, 3


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


def main():
    parser = argparse.ArgumentParser(description="Treinador BTD6 com Memória Persistente")
    parser.add_argument("--difficulty", type=str, default="easy", choices=["easy", "medium", "hard", "chimps"])
    parser.add_argument("--generations", type=int, default=0, help="0 = Roda continuamente até Ctrl+C")
    parser.add_argument("--pop-size", type=int, default=512)
    parser.add_argument("--genes", type=int, default=90)
    parser.add_argument("--reset-memory", action="store_true", help="Apaga a memória anterior e começa do zero")
    args = parser.parse_args()

    cfg = DIFFICULTY_CONFIG[args.difficulty]
    target_round = cfg["target_round"]
    vault = AIMemoryVault()

    if args.reset_memory:
        population = create_random_population(args.pop_size, args.genes, target_round, MONKEY_MEADOW_PATH)
        gen = 1
        best_global_fitness = 0.0
        total_episodes = 0
        print(f"[MEMÓRIA] Memória reiniciada manualmente para o modo {args.difficulty.upper()}.")
    else:
        population, meta = vault.load_state(
            args.difficulty, args.pop_size, args.genes, target_round, create_random_population, MONKEY_MEADOW_PATH
        )
        gen = int(meta.get("generation", 1))
        best_global_fitness = float(meta.get("best_fitness", 0.0))
        total_episodes = int(meta.get("total_episodes_learned", 0))

        if meta.get("resumed_from_memory"):
            if meta.get("transfer_source"):
                print(f"[MEMÓRIA] Transfer Learning ativo! Aprendizado importado do modo {meta['transfer_source']}.")
            else:
                print(
                    f"[MEMÓRIA] Aprendizado restaurado! Retomando da Geração #{gen} "
                    f"| Recorde salvo: {best_global_fitness:,.1f} ({total_episodes:,} partidas na memória)"
                )
        else:
            print(f"[MEMÓRIA] Nenhum histórico anterior encontrado para {args.difficulty.upper()}. Iniciando Geração #1.")

    # Warm-up do Numba JIT
    _ = evaluate_population_parallel(
        population[:4], target_round, cfg["start_cash"], cfg["start_lives"], cfg["cost_mult"], MONKEY_MEADOW_PATH
    )

    stagnation_counter = 0
    base_mutation_rate = 0.16
    last_fitnesses = np.zeros(args.pop_size, dtype=np.float64)
    last_rounds = np.zeros(args.pop_size, dtype=np.int32)
    last_cashes = np.zeros(args.pop_size, dtype=np.float32)
    last_lives = np.zeros(args.pop_size, dtype=np.int32)

    try:
        while args.generations == 0 or gen <= args.generations:
            t0 = time.perf_counter()
            fitnesses, rounds, cashes, lives_arr = evaluate_population_parallel(
                population, target_round, cfg["start_cash"], cfg["start_lives"], cfg["cost_mult"], MONKEY_MEADOW_PATH
            )
            dt = max(1e-6, time.perf_counter() - t0)
            epm = int((args.pop_size / dt) * 60.0)
            total_episodes += args.pop_size

            last_fitnesses, last_rounds, last_cashes, last_lives = fitnesses, rounds, cashes, lives_arr

            best_idx = int(np.argmax(fitnesses))
            gen_max_fit = float(fitnesses[best_idx])
            gen_mean_fit = float(np.mean(fitnesses))
            best_round = int(rounds[best_idx])
            best_lives = int(lives_arr[best_idx])
            best_cash = float(cashes[best_idx])

            saved_tag = ""
            # Salva na memória sempre que bater recorde OU a cada 10 gerações
            if gen_max_fit > best_global_fitness + 1e-3:
                best_global_fitness = gen_max_fit
                stagnation_counter = 0
                vault.save_state(
                    args.difficulty, population, fitnesses, gen, best_global_fitness,
                    best_round, best_cash, best_lives, target_round, total_episodes
                )
                saved_tag = " [MEMÓRIA ATUALIZADA - NOVO RECORDE!]"
            elif gen % 10 == 0:
                stagnation_counter += 1
                vault.save_state(
                    args.difficulty, population, fitnesses, gen, best_global_fitness,
                    best_round, best_cash, best_lives, target_round, total_episodes
                )
                saved_tag = " [CHECKPOINT AUTO-SALVO]"
            else:
                stagnation_counter += 1

            current_mut = min(0.45, base_mutation_rate + stagnation_counter * 0.025) if stagnation_counter >= 5 else base_mutation_rate
            v_str = "PASSOU!" if (best_round >= target_round and best_lives > 0) else f"R{best_round:02d}/{target_round}"

            print(
                f"Gen {gen:04d} | Max: {gen_max_fit:10,.1f} | Recorde: {best_global_fitness:10,.1f} | "
                f"Mean: {gen_mean_fit:9,.1f} | {v_str} (♥ {best_lives:3d}) | EPM: {epm:6d}{saved_tag}"
            )

            population = evolve_next_generation(population, fitnesses, current_mut, target_round, MONKEY_MEADOW_PATH)
            gen += 1

    except KeyboardInterrupt:
        print("\n[MEMÓRIA] Interrupção detectada (Ctrl+C)! Salvando aprendizado acumulado antes de fechar...")
        if np.any(last_fitnesses > 0):
            b_idx = int(np.argmax(last_fitnesses))
            vault.save_state(
                args.difficulty, population, last_fitnesses, gen, best_global_fitness,
                int(last_rounds[b_idx]), float(last_cashes[b_idx]), int(last_lives[b_idx]), target_round, total_episodes
            )
        print("[MEMÓRIA] Estado salvo com sucesso em memory/. Na próxima execução ela continuará daqui!")


if __name__ == "__main__":
    main()