import os
import json
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple


class AIMemoryVault:
    """
    Sistema de Memória Persistente para o Simulador BTD6.
    Salva e restaura: População inteira, Hall da Fama (Elites), Geração atual,
    Histórico de Fitness e Replays, além de suportar Transfer Learning entre dificuldades.
    """

    def __init__(self, base_dir: str = "memory"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def _get_vault_path(self, difficulty: str) -> str:
        return os.path.join(self.base_dir, f"btd6_memory_{difficulty.lower()}.npz")

    def _get_meta_path(self, difficulty: str) -> str:
        return os.path.join(self.base_dir, f"btd6_meta_{difficulty.lower()}.json")

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
        total_episodes: int = 0,
    ) -> str:
        """
        Salva toda a memória de aprendizado da IA de forma atômica em disco.
        Mantém um Hall da Fama cumulativo com os 32 melhores cérebros da história.
        """
        difficulty = difficulty.lower()
        vault_path = self._get_vault_path(difficulty)
        meta_path = self._get_meta_path(difficulty)

        # Ordena população atual do melhor para o pior
        sorted_idx = np.argsort(fitnesses)[::-1]
        sorted_pop = population[sorted_idx].astype(np.float32)
        sorted_fit = fitnesses[sorted_idx].astype(np.float64)

        # Mescla com o Hall da Fama anterior se já existir na memória
        hof_size = min(32, len(sorted_pop))
        new_hof_pop = sorted_pop[:hof_size].copy()
        new_hof_fit = sorted_fit[:hof_size].copy()

        if os.path.exists(vault_path):
            try:
                with np.load(vault_path) as data:
                    if "hall_of_fame" in data and "hof_fitness" in data:
                        old_hof = data["hall_of_fame"]
                        old_fit = data["hof_fitness"]
                        if old_hof.shape[1:] == new_hof_pop.shape[1:]:
                            comb_pop = np.concatenate([old_hof, new_hof_pop], axis=0)
                            comb_fit = np.concatenate([old_fit, new_hof_fit], axis=0)
                            # Filtra indivíduos únicos pelo valor de fitness para evitar clones puros
                            _, unique_indices = np.unique(np.round(comb_fit, 2), return_index=True)
                            top_u = unique_indices[np.argsort(comb_fit[unique_indices])[::-1]][:hof_size]
                            if len(top_u) == hof_size:
                                new_hof_pop = comb_pop[top_u]
                                new_hof_fit = comb_fit[top_u]
            except Exception:
                pass

        best_chrom = new_hof_pop[0]
        if float(new_hof_fit[0]) > best_fitness:
            best_fitness = float(new_hof_fit[0])

        # Escrita Atômica do arquivo binário .npz
        tmp_vault = vault_path + ".tmp.npz"
        np.savez_compressed(
            tmp_vault,
            population=sorted_pop,
            fitnesses=sorted_fit,
            hall_of_fame=new_hof_pop,
            hof_fitness=new_hof_fit,
            best_chrom=best_chrom,
        )
        os.replace(tmp_vault, vault_path)

        # Escrita Atômica dos metadados legíveis (.json)
        meta_payload = {
            "difficulty": difficulty,
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
        tmp_meta = meta_path + ".tmp"
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp_meta, meta_path)

        # Atualiza também o replay do campeão para o visualizador
        self._export_champion_replay(best_chrom, meta_payload)
        return vault_path

    def load_state(
        self,
        difficulty: str,
        desired_pop_size: int,
        n_genes: int,
        target_round: int,
        fallback_creator_fn,
        path_coords: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Carrega a memória salva da dificuldade escolhida.
        Se não houver memória para a dificuldade atual, tenta Transfer Learning
        (importando o aprendizado de dificuldades anteriores como base).
        """
        difficulty = difficulty.lower()
        vault_path = self._get_vault_path(difficulty)
        meta_path = self._get_meta_path(difficulty)

        if os.path.exists(vault_path) and os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                with np.load(vault_path) as data:
                    saved_pop = data["population"].astype(np.float32)
                    hof_pop = data["hall_of_fame"].astype(np.float32)

                pop = self._adapt_population_shape(
                    saved_pop, hof_pop, desired_pop_size, n_genes, target_round, fallback_creator_fn, path_coords
                )
                meta["resumed_from_memory"] = True
                meta["transfer_source"] = None
                return pop, meta
            except Exception as e:
                print(f"[MEMÓRIA] Aviso ao ler cofre ({e}). Criando nova população.")

        # Tentativa de Transfer Learning (Ex: aprendeu no Easy -> usa no Medium/Hard)
        hierarchy = ["easy", "medium", "hard", "chimps"]
        if difficulty in hierarchy:
            curr_idx = hierarchy.index(difficulty)
            for prev_idx in range(curr_idx - 1, -1, -1):
                prev_diff = hierarchy[prev_idx]
                prev_vault = self._get_vault_path(prev_diff)
                if os.path.exists(prev_vault):
                    try:
                        with np.load(prev_vault) as data:
                            hof_pop = data["hall_of_fame"].astype(np.float32)
                        pop = self._adapt_population_shape(
                            hof_pop, hof_pop, desired_pop_size, n_genes, target_round, fallback_creator_fn, path_coords
                        )
                        meta = {
                            "difficulty": difficulty,
                            "generation": 1,
                            "total_episodes_learned": 0,
                            "best_fitness": 0.0,
                            "best_round": 0,
                            "resumed_from_memory": True,
                            "transfer_source": prev_diff.upper(),
                        }
                        return pop, meta
                    except Exception:
                        pass

        # Caso seja a primeira execução absoluta do projeto
        pop = fallback_creator_fn(desired_pop_size, n_genes, target_round, path_coords)
        meta = {
            "difficulty": difficulty,
            "generation": 1,
            "total_episodes_learned": 0,
            "best_fitness": 0.0,
            "best_round": 0,
            "resumed_from_memory": False,
            "transfer_source": None,
        }
        return pop, meta

    def _adapt_population_shape(
        self,
        saved_pop: np.ndarray,
        hof_pop: np.ndarray,
        desired_pop_size: int,
        n_genes: int,
        target_round: int,
        fallback_creator_fn,
        path_coords: np.ndarray,
    ) -> np.ndarray:
        """Ajusta o tamanho da população caso o usuário mude o Batch/Pop Size na interface."""
        new_pop = fallback_creator_fn(desired_pop_size, n_genes, target_round, path_coords)
        min_genes = min(n_genes, saved_pop.shape[1])

        # Insere o Hall da Fama intacto no topo da população
        n_hof = min(len(hof_pop), desired_pop_size // 4)
        for i in range(n_hof):
            new_pop[i, :min_genes, :] = hof_pop[i, :min_genes, :]

        # Preenche o restante com a população salva
        n_copy = min(len(saved_pop), desired_pop_size - n_hof)
        for i in range(n_copy):
            new_pop[n_hof + i, :min_genes, :] = saved_pop[i, :min_genes, :]

        return new_pop

    def _export_champion_replay(self, best_chrom: np.ndarray, meta: Dict[str, Any]):
        actions = []
        for g in range(best_chrom.shape[0]):
            act_id = int(best_chrom[g, 0])
            if act_id == 0:
                continue
            act_name = {1: "PLACE", 2: "UPGRADE", 3: "SELL"}.get(act_id, "NONE")
            actions.append({
                "action": act_name,
                "round": int(best_chrom[g, 1]),
                "slot_or_type": int(best_chrom[g, 2]),
                "x": round(float(best_chrom[g, 3]), 1),
                "y": round(float(best_chrom[g, 4]), 1),
                "upgrade_path": int(best_chrom[g, 5]) % 3,
            })

        payload = dict(meta)
        payload["actions"] = actions

        for fname in ["champion_replay.json", f"champion_{meta['difficulty']}.json"]:
            fpath = os.path.join("checkpoints", fname)
            tmp_f = fpath + ".tmp"
            with open(tmp_f, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            os.replace(tmp_f, fpath)