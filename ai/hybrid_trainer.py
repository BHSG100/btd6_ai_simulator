import os
import json
import numpy as np
from ai.smart_env import TOWER_CATALOG


class AdaptiveGeneticOptimizer:
    """Otimizador Genético com Elitismo, Crossover Uniforme e Hiper-Mutação anti-estagnação."""

    def __init__(self, pop_size=12, num_anchors=24, save_path="memory/best_genome.json"):
        self.pop_size = pop_size
        self.num_anchors = num_anchors
        self.num_tower_types = len(TOWER_CATALOG)
        self.save_path = save_path
        self.generation = 1
        self.best_fitness = -1e9
        self.best_round = 1
        self.stagnation_counter = 0
        self.fitness_history = []

        # Cada indivíduo possui uma ordem de prioridade dos 24 anchors + tipo de torre + modo de alvo
        self.pop_order = np.zeros((pop_size, num_anchors), dtype=np.int32)
        self.pop_towers = np.zeros((pop_size, num_anchors), dtype=np.int32)
        self.pop_targets = np.zeros((pop_size, num_anchors), dtype=np.int32)
        self.best_genome = None

        self._init_population()
        self._load_checkpoint()

    def _init_population(self):
        for i in range(self.pop_size):
            # Prioriza os primeiros anchors (maior score de cobertura) com permutação suave
            base_order = np.arange(self.num_anchors, dtype=np.int32)
            np.random.shuffle(base_order[:12])
            self.pop_order[i] = base_order
            self.pop_towers[i] = np.random.randint(0, self.num_tower_types, size=self.num_anchors)
            # Garante equilíbrio inicial (Alch/Village + DPS) nos 4 primeiros slots
            self.pop_towers[i, 0] = 2  # Ninja 4-0-2
            self.pop_towers[i, 1] = 3  # Alch 4-2-0
            self.pop_targets[i] = np.random.choice([0, 3], size=self.num_anchors)

    def select_action_for_arena(self, idx, env):
        """Escolhe a próxima compra da fila genética assim que houver ouro disponível."""
        cash = env.game_state[0]
        order = self.pop_order[idx]
        towers = self.pop_towers[idx]

        for step_slot in range(self.num_anchors):
            a_idx = order[step_slot]
            if env.t_active[a_idx] == 0:
                t_id = towers[step_slot]
                cost = TOWER_CATALOG[t_id][1]
                if cash >= cost:
                    return 1 + a_idx * self.num_tower_types + t_id
                else:
                    # Aguarda juntar dinheiro para a próxima torre planejada da build
                    return 0
        return 0

    def evolve(self, fitness_scores, rounds_reached):
        best_idx = int(np.argmax(fitness_scores))
        gen_max_fit = float(fitness_scores[best_idx])
        gen_max_round = int(rounds_reached[best_idx])
        self.fitness_history.append(gen_max_fit)

        if gen_max_fit > self.best_fitness:
            self.best_fitness = gen_max_fit
            self.best_round = max(self.best_round, gen_max_round)
            self.stagnation_counter = 0
            self.best_genome = {
                "generation": self.generation,
                "best_fitness": self.best_fitness,
                "best_round": self.best_round,
                "order": self.pop_order[best_idx].tolist(),
                "towers": self.pop_towers[best_idx].tolist(),
                "targets": self.pop_targets[best_idx].tolist(),
            }
            self._save_checkpoint()
        else:
            self.stagnation_counter += 1

        # Taxa de mutação adaptativa: sobe se estagnar por mais de 3 gerações
        mut_rate = min(0.42, 0.12 + self.stagnation_counter * 0.05)

        ranked = np.argsort(fitness_scores)[::-1]
        new_order = np.zeros_like(self.pop_order)
        new_towers = np.zeros_like(self.pop_towers)
        new_targets = np.zeros_like(self.pop_targets)

        # Elitismo dos 2 melhores
        for e in range(2):
            new_order[e] = self.pop_order[ranked[e]].copy()
            new_towers[e] = self.pop_towers[ranked[e]].copy()
            new_targets[e] = self.pop_targets[ranked[e]].copy()

        # Se tivermos campeão global salvo, garante que o slot 0 nunca regrida
        if self.best_genome is not None:
            new_order[0] = np.array(self.best_genome["order"], dtype=np.int32)
            new_towers[0] = np.array(self.best_genome["towers"], dtype=np.int32)
            new_targets[0] = np.array(self.best_genome["targets"], dtype=np.int32)

        # Crossover Uniforme + Mutação de Ordem e Tipo
        for i in range(2, self.pop_size):
            p1, p2 = np.random.choice(ranked[:6], size=2, replace=False)
            mask = np.random.rand(self.num_anchors) < 0.5
            child_towers = np.where(mask, self.pop_towers[p1], self.pop_towers[p2])
            child_targets = np.where(mask, self.pop_targets[p1], self.pop_targets[p2])
            child_order = self.pop_order[p1].copy()

            # Mutação de genes de torres
            for g in range(self.num_anchors):
                if np.random.rand() < mut_rate:
                    child_towers[g] = np.random.randint(0, self.num_tower_types)

            # Mutação por swap na ordem dos slots táticos
            if np.random.rand() < mut_rate * 1.5:
                s1, s2 = np.random.randint(0, 14, size=2)
                child_order[s1], child_order[s2] = child_order[s2], child_order[s1]

            new_order[i] = child_order
            new_towers[i] = child_towers
            new_targets[i] = child_targets

        self.pop_order = new_order
        self.pop_towers = new_towers
        self.pop_targets = new_targets
        self.generation += 1

    def _save_checkpoint(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(self.best_genome, f, indent=2)

    def _load_checkpoint(self):
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.best_genome = data
                self.best_fitness = float(data.get("best_fitness", -1e9))
                self.best_round = int(data.get("best_round", 1))
                self.generation = int(data.get("generation", 1))
                self.pop_order[0] = np.array(data["order"], dtype=np.int32)
                self.pop_towers[0] = np.array(data["towers"], dtype=np.int32)
                self.pop_targets[0] = np.array(data["targets"], dtype=np.int32)
            except Exception:
                pass