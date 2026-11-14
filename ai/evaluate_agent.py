from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from ai.ga_evaluator import GeneOp
from ai.gym_env import BTD6GymEnv
from core.arenas import TargetingMode
from core.engine import GameEngine
from core.grid import CellState
from utils.data_loader import GameDataLoader

def render_ascii(engine):
    w, h = engine.grid_width, engine.grid_height
    canvas = np.full((h, w), ".", dtype="<U2")
    for x in range(w):
        for y in range(h):
            if engine.grid.grid[x, y] == CellState.PATH: canvas[y, x] = "#"
    for slot in np.flatnonzero(engine.towers.arena.active):
        gx, gy = engine.towers.grid_coords[int(slot)]
        canvas[gy, gx] = str(slot) if slot < 10 else chr(ord('A') + (slot-10)%26)
    
    out = ["  " + "".join(f"{x%10}" for x in range(w))]
    for y in range(h):
        out.append(f"{y:2d} " + "".join(canvas[y]))
    return "\n".join(out)

def replay_ga(engine, genome):
    engine.reset()
    tids = sorted(engine.towers_catalog.keys())
    g = genome[np.argsort(genome[:, 0])] if genome.size > 0 else []
    cursor, n = 0, len(g)
    
    while not engine.is_game_over and not engine.is_victory:
        r_idx = engine.current_round_idx
        while cursor < n and g[cursor, 0] <= r_idx:
            gen = g[cursor]
            op = gen[1]
            if op == GeneOp.PLACE:
                engine.action_place_tower(tids[gen[2]], gen[3], gen[4], target_mode=TargetingMode(np.clip(gen[5],0,3)))
            elif op == GeneOp.UPGRADE:
                engine.action_upgrade_tower(gen[2], gen[3])
            elif op == GeneOp.SELL:
                engine.action_sell_tower(gen[2])
            cursor += 1
        res = engine.run_round()
        print(f"Round {r_idx+1}: Pops {res.pops} | Cash {engine.cash} | Lives {engine.lives}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "ga", "ppo"], default="demo")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = json.loads((root / "data" / "game_config.json").read_text(encoding="utf-8"))
    
    tdb = GameDataLoader.load_towers(root / "data" / "towers.json")
    bdb, blup = GameDataLoader.load_bloons(root / "data" / "bloons.json")
    rsched = GameDataLoader.load_and_compile_rounds(root / "data" / "rounds.json", bdb)

    mm = cfg["maps"]["monkey_meadow"]
    gw, gh = mm["grid_width"], mm["grid_height"]
    wp = np.array(mm["waypoints"], dtype=np.float32)
    pc = np.array(mm["path_cells"], dtype=np.int32)

    engine = GameEngine(gw, gh, wp, pc, tdb, blup, rsched, starting_cash=3500)
    
    if args.mode == "demo":
        print("Rodando Demo Warm-Start...")
        g = np.array([
            [0, 0, 2, 10, 7, 0],  # Place Dart
            [1, 0, 1, 14, 7, 3],  # Place Bomb
            [3, 1, 0, 0, -1, 0],  # Upgrade Dart Top
            [3, 1, 0, 0, -1, 0],  # Upgrade Dart Top
            [5, 0, 0, 12, 7, 0],  # Place Alch
        ], dtype=np.int32)
        replay_ga(engine, g)
        print("\nMAPA ASCII:")
        print(render_ascii(engine))
    else:
        print(f"Modo {args.mode} requer treinos prévios gerados em checkpoints/.")

if __name__ == "__main__":
    main()
