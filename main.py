import time
import argparse
import numpy as np
from ai.smart_env import BTD6TacticalEnv
from ai.hybrid_trainer import AdaptiveGeneticOptimizer
from ui.visualizer import CyberArenaVisualizer


def run_visual_training(num_arenas=12, max_rounds=80):
    print("[INIT] Compilando Kernels Numba SoA (Warm-up JIT)...")
    envs = [BTD6TacticalEnv(max_rounds=max_rounds) for _ in range(num_arenas)]
    # Dispara 1 passo para compilar o código nativo antes de abrir a janela
    envs[0].step(0)

    optimizer = AdaptiveGeneticOptimizer(pop_size=num_arenas, num_anchors=envs[0].max_towers)
    vis = CyberArenaVisualizer(num_arenas=num_arenas)

    fitness_scores = np.zeros(num_arenas, dtype=np.float32)
    rounds_reached = np.ones(num_arenas, dtype=np.int32)
    dones = np.zeros(num_arenas, dtype=bool)

    running = True
    while running:
        running = vis.handle_events()
        if not vis.paused:
            t0 = time.perf_counter()
            ticks_executed = 0

            for _ in range(vis.speed_multiplier):
                for i, env in enumerate(envs):
                    if dones[i]:
                        continue
                    act = optimizer.select_action_for_arena(i, env)
                    _, rew, term, trunc, _ = env.step(act)
                    fitness_scores[i] += rew
                    rounds_reached[i] = max(rounds_reached[i], env.current_round)
                    ticks_executed += 30
                    if term or trunc:
                        dones[i] = True

                if np.all(dones):
                    optimizer.evolve(fitness_scores, rounds_reached)
                    for env in envs:
                        env.reset()
                    fitness_scores.fill(0.0)
                    rounds_reached.fill(1)
                    dones.fill(False)
                    break

            dt_wall = max(1e-6, time.perf_counter() - t0)
            tps = ticks_executed / dt_wall
        else:
            tps = 0.0

        vis.render(envs, fitness_scores, optimizer, tps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTD6 AI Simulator v2.0 Pro")
    parser.add_argument("--arenas", type=int, default=12, help="Número de arenas simultâneas")
    parser.add_argument("--rounds", type=int, default=80, help="Ronda alvo máxima")
    args = parser.parse_args()
    run_visual_training(num_arenas=args.arenas, max_rounds=args.rounds)