from __future__ import annotations
import os
from pathlib import Path
import json
import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from ai.gym_env import BTD6GymEnv
from utils.data_loader import GameDataLoader

def _mask_fn(env: gym.Env):
    return env.unwrapped.action_masks()

def make_env(gw, gh, wp, pc, tdb, bl, rs, rank):
    def _init():
        base = BTD6GymEnv(gw, gh, wp, pc, tdb, bl, rs, max_towers=32, dt=0.05)
        base.reset(seed=42+rank)
        return ActionMasker(base, _mask_fn)
    return _init

def main():
    root = Path(__file__).resolve().parent.parent
    cfg = json.loads((root / "data" / "game_config.json").read_text(encoding="utf-8"))
    
    towers_db = GameDataLoader.load_towers(root / "data" / "towers.json")
    bloons_db, bloon_lookup = GameDataLoader.load_bloons(root / "data" / "bloons.json")
    rounds_schedule = GameDataLoader.load_and_compile_rounds(root / "data" / "rounds.json", bloons_db)

    mm = cfg["maps"]["monkey_meadow"]
    gw, gh = mm["grid_width"], mm["grid_height"]
    wp = np.array(mm["waypoints"], dtype=np.float32)
    pc = np.array(mm["path_cells"], dtype=np.int32)

    cores = max(2, (os.cpu_count() or 4) - 1)
    model_dir = root / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)

    train_env = VecMonitor(SubprocVecEnv([make_env(gw, gh, wp, pc, towers_db, bloon_lookup, rounds_schedule, i) for i in range(cores)]))
    eval_env = VecMonitor(SubprocVecEnv([make_env(gw, gh, wp, pc, towers_db, bloon_lookup, rounds_schedule, 999)]))

    cb = MaskableEvalCallback(eval_env, best_model_save_path=str(model_dir/"best_ppo"), eval_freq=2000, n_eval_episodes=5, deterministic=True)
    
    model = MaskablePPO("MlpPolicy", train_env, learning_rate=3e-4, n_steps=512, batch_size=256, n_epochs=10, 
                        policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}}, verbose=1)
    
    print(f"Treinando PPO em {cores} envs...")
    model.learn(total_timesteps=100000, callback=cb, progress_bar=True)
    model.save(str(model_dir / "btd6_maskable_ppo_final"))

if __name__ == "__main__":
    main()
