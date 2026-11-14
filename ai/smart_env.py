import numpy as np
import gymnasium as gym
from gymnasium import spaces
from core.engine import (
    step_macro_ticks_jit,
    DMG_SHARP,
    DMG_EXPLOSION,
    DMG_ENERGY,
    DMG_NORMAL,
    TARGET_FIRST,
    TARGET_STRONG,
)


def compute_tactical_anchors(wp_coords, num_anchors=24, width=800, height=600, min_track_dist=28.0):
    """Encontra automaticamente os melhores chokepoints do mapa maximizando cobertura de trilha."""
    # Amostra pontos densos na trilha
    dense_track = []
    for i in range(len(wp_coords) - 1):
        p1, p2 = wp_coords[i], wp_coords[i + 1]
        steps = max(2, int(np.linalg.norm(p2 - p1) // 8))
        for s in range(steps):
            dense_track.append(p1 + (p2 - p1) * (s / steps))
    dense_track = np.array(dense_track, dtype=np.float32)

    candidates = []
    for gx in range(60, width - 60, 28):
        for gy in range(60, height - 60, 28):
            pt = np.array([gx, gy], dtype=np.float32)
            dists = np.linalg.norm(dense_track - pt, axis=1)
            if np.min(dists) < min_track_dist:
                continue  # Em cima da trilha (inválido)
            # Score: pontos da trilha dentro de raio médio (135px)
            coverage_score = np.sum(dists <= 135.0) + 0.5 * np.sum(dists <= 90.0)
            if coverage_score > 5:
                candidates.append((coverage_score, gx, gy))

    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = []
    for score, gx, gy in candidates:
        if all((gx - sx) ** 2 + (gy - sy) ** 2 >= 34.0**2 for sx, sy in selected):
            selected.append((gx, gy))
            if len(selected) >= num_anchors:
                break
    return np.array(selected, dtype=np.float32)


# Catálogo de Torres Balanceado com Sinergia Oficial e Custos Reais
TOWER_CATALOG = [
    # nome, custo, range, dmg, moab_bonus, pierce, cd, camo, dmg_type, splash, is_alch, is_vil, income
    ("Dart 0-2-3 (Crossbow)", 1150, 155.0, 3.0, 0.0, 4, 0.45, 1, DMG_SHARP, 0.0, 0, 0, 0.0),
    ("Bomb 2-0-4 (Recursive)", 3800, 125.0, 4.0, 2.0, 28, 1.10, 0, DMG_EXPLOSION, 52.0, 0, 0, 0.0),
    ("Ninja 4-0-2 (Bloonjitsu)", 4650, 130.0, 2.0, 1.0, 5, 0.14, 1, DMG_SHARP, 0.0, 0, 0, 0.0),
    ("Alch 4-2-0 (Strong Stim)", 4800, 115.0, 2.0, 0.0, 15, 1.60, 0, DMG_NORMAL, 25.0, 1, 0, 0.0),
    ("Super 2-0-3 (Dark Knight)", 9200, 165.0, 3.0, 8.0, 4, 0.055, 1, DMG_ENERGY, 0.0, 0, 0, 0.0),
    ("Village 2-3-0 (MIB Radar)", 6500, 145.0, 1.0, 0.0, 0, 99.0, 1, DMG_NORMAL, 0.0, 0, 1, 0.0),
    ("Farm 2-3-0 (Monkey Bank)", 4900, 60.0, 0.0, 0.0, 0, 99.0, 0, DMG_SHARP, 0.0, 0, 0, 420.0),
    ("Sniper 4-0-2 (Maim MOAB)", 6800, 999.0, 18.0, 35.0, 2, 1.35, 1, DMG_NORMAL, 0.0, 0, 0, 0.0),
]

class SmartEnv:
    def __init__(self, starting_cash=650.0):
        self.towers = []
        self.cash = starting_cash
        self.tier5_registry = set() # Trava de Tier 5 único (ex: "DartMonkey_path0")

    def upgrade_tower(self, tower_idx, path, cost):
        """
        Gere a evolução 5-2-0 a partir do Tier 0 e limite global de Tier 5.
        """
        tower = self.towers[tower_idx]
        
        # 1. Verifica saldo
        if self.cash < cost:
            return False, "Not enough cash"

        # 2. Lógica Oficial de Crosspathing 5-2-0
        active_paths = sum(1 for t in tower['tiers'] if t > 0)
        
        # Bloqueia o terceiro caminho
        if tower['tiers'][path] == 0 and active_paths >= 2:
            return False, "Crosspath locked (Max 2 paths)"
            
        # Bloqueia upar o secundário além do Tier 2 se já houver um Tier 3+
        main_path = -1
        for i, t in enumerate(tower['tiers']):
            if t >= 3:
                main_path = i
                
        if main_path != -1 and main_path != path:
            if tower['tiers'][path] >= 2:
                return False, "Secondary path capped at Tier 2"
                
        # 3. Trava Global de Tier 5
        if tower['tiers'][path] == 4:
            t5_id = f"{tower['type']}_path{path}"
            if t5_id in self.tier5_registry:
                return False, "Tier 5 already exists"
            self.tier5_registry.add(t5_id)

        # 4. Aplica o Upgrade
        self.cash -= cost
        tower['tiers'][path] += 1
        tower['total_invested'] += cost
        return True, "Upgrade successful"

    def sell_tower(self, tower_idx):
        """
        Venda com Reembolso de 70% e liberação de Tiers 5.
        """
        if tower_idx >= len(self.towers):
            return 0
            
        tower = self.towers.pop(tower_idx)
        
        # Reembolso de 70%
        refund_amount = tower['total_invested'] * 0.70
        self.cash += refund_amount
        
        # Se a torre vendida era Tier 5, libera a construção novamente
        for i, t in enumerate(tower['tiers']):
            if t == 5:
                t5_id = f"{tower['type']}_path{i}"
                self.tier5_registry.discard(t5_id)
                
        return refund_amount
class BTD6TacticalEnv(gym.Env):
    """Ambiente v2.4 com Árvore 5-2-0, Venda 70%, Rondas 1-100 (DDT/BAD), Modo CHIMPS e Previsão."""

    def __init__(self, max_rounds=100, chimps_mode=False):
        super().__init__()
        self.chimps_mode = chimps_mode
        self.max_rounds = 100 if chimps_mode else max_rounds
        self.max_bloons = 512
        self.max_towers = 24
        self.max_lasers = 64
        # Trilha estilo Monkey Meadow com curvas estratégicas
        self.wp_coords = np.array(
            [
                [0.0, 240.0],
                [260.0, 240.0],
                [260.0, 110.0],
                [480.0, 110.0],
                [480.0, 440.0],
                [200.0, 440.0],
                [200.0, 340.0],
                [660.0, 340.0],
                [660.0, 180.0],
                [800.0, 180.0],
            ],
            dtype=np.float32,
        )
        diffs = np.diff(self.wp_coords, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        self.wp_cum_dist = np.concatenate(([0.0], np.cumsum(seg_lens))).astype(np.float32)
        self.total_track_len = float(self.wp_cum_dist[-1])

        self.anchors = compute_tactical_anchors(self.wp_coords, num_anchors=self.max_towers)
        self.num_tower_types = len(TOWER_CATALOG)

        # Espaço de Ações Discreto:
        # 0: Avançar Tempo (Macro Wait)
        # 1 .. (24 * 8): Construir/Upgrade Torre T no Anchor A
        # (24 * 8 + 1) .. (24 * 9): Alternar Alvo (First <-> Strong) no Anchor A
        self.total_actions = 1 + self.max_towers * self.num_tower_types + self.max_towers
        self.action_space = spaces.Discrete(self.total_actions)
        self.observation_space = spaces.Box(
            low=-1.0, high=10.0, shape=(6 + self.max_towers * 4,), dtype=np.float32
        )

        self._allocate_soa_buffers()
        self.reset()

    def _allocate_soa_buffers(self):
        self.b_active = np.zeros(self.max_bloons, dtype=np.int32)
        self.b_hp = np.zeros(self.max_bloons, dtype=np.float32)
        self.b_max_hp = np.zeros(self.max_bloons, dtype=np.float32)
        self.b_speed = np.zeros(self.max_bloons, dtype=np.float32)
        self.b_dist = np.zeros(self.max_bloons, dtype=np.float32)
        self.b_x = np.zeros(self.max_bloons, dtype=np.float32)
        self.b_y = np.zeros(self.max_bloons, dtype=np.float32)
        self.b_camo = np.zeros(self.max_bloons, dtype=np.int32)
        self.b_immun = np.zeros(self.max_bloons, dtype=np.int32)
        self.b_rbe = np.zeros(self.max_bloons, dtype=np.float32)
        self.b_children_count = np.zeros(self.max_bloons, dtype=np.int32)
        self.b_child_hp = np.zeros(self.max_bloons, dtype=np.float32)
        self.b_child_speed = np.zeros(self.max_bloons, dtype=np.float32)
        self.b_child_immun = np.zeros(self.max_bloons, dtype=np.int32)
        self.b_is_moab = np.zeros(self.max_bloons, dtype=np.int32)

        self.t_active = np.zeros(self.max_towers, dtype=np.int32)
        self.t_type_id = np.full(self.max_towers, -1, dtype=np.int32)
        self.t_x = self.anchors[:, 0].copy()
        self.t_y = self.anchors[:, 1].copy()
        self.t_range = np.zeros(self.max_towers, dtype=np.float32)
        self.t_damage = np.zeros(self.max_towers, dtype=np.float32)
        self.t_moab_bonus = np.zeros(self.max_towers, dtype=np.float32)
        self.t_pierce = np.zeros(self.max_towers, dtype=np.int32)
        self.t_cooldown = np.zeros(self.max_towers, dtype=np.float32)
        self.t_timer = np.zeros(self.max_towers, dtype=np.float32)
        self.t_camo_see = np.zeros(self.max_towers, dtype=np.int32)
        self.t_dmg_type = np.zeros(self.max_towers, dtype=np.int32)
        self.t_splash_radius = np.zeros(self.max_towers, dtype=np.float32)
        self.t_target_mode = np.zeros(self.max_towers, dtype=np.int32)
        self.t_is_alch = np.zeros(self.max_towers, dtype=np.int32)
        self.t_is_village = np.zeros(self.max_towers, dtype=np.int32)
        self.t_income = np.zeros(self.max_towers, dtype=np.float32)
        self.t_pops = np.zeros(self.max_towers, dtype=np.float32)

        self.laser_buf = np.zeros((self.max_lasers, 5), dtype=np.float32)
        self.laser_count_arr = np.zeros(1, dtype=np.int32)
        self.spawn_cursor_arr = np.zeros(1, dtype=np.int32)
        self.game_state = np.zeros(4, dtype=np.float32)

    def _build_round_wave(self, round_num):
        """Gera ondas fiéis às curvas de dificuldade do BTD6 (Camo 24, Lead 28, MOAB 40, Ceramic Rush 63/78)."""
        count = 12 + round_num * 3
        if round_num in (63, 76, 78):
            count = 95  # Rush cerâmico intenso

        ticks = np.arange(count, dtype=np.int32) * max(3, 22 - round_num // 4)
        hp = np.full(count, 1.0 + (round_num // 5) * 1.5, dtype=np.float32)
        speed = np.full(count, 65.0 + round_num * 1.4, dtype=np.float32)
        camo = np.zeros(count, dtype=np.int32)
        immun = np.zeros(count, dtype=np.int32)
        rbe = hp * 2.0
        children = np.zeros(count, dtype=np.int32)
        child_hp = np.zeros(count, dtype=np.float32)
        child_speed = np.zeros(count, dtype=np.float32)
        child_immun = np.zeros(count, dtype=np.int32)
        is_moab = np.zeros(count, dtype=np.int32)

        if round_num >= 24 and round_num % 3 == 0:
            camo[::2] = 1
        if round_num >= 28 and round_num % 4 == 0:
            immun[::3] = DMG_SHARP  # Lead Bloons
            hp[::3] += 4.0
        if round_num >= 25 and round_num % 5 == 0:
            immun[1::4] = DMG_ENERGY  # Purple Bloons
            speed[1::4] *= 1.45

        if round_num >= 38:
            # Ceramics que geram 2 Rainbows ao estourar
            cer_mask = np.arange(count) % 4 == 0
            hp[cer_mask] = 10.0 + (round_num - 38) * 0.8
            children[cer_mask] = 2
            child_hp[cer_mask] = 5.0
            child_speed[cer_mask] = speed[cer_mask] * 1.25
            rbe[cer_mask] = hp[cer_mask] + 10.0

        if round_num >= 40 and (round_num % 5 == 0 or round_num >= 60):
            # MOABs / BFBs / ZOMGs
            num_moabs = 1 + max(0, (round_num - 40) // 8)
            for m in range(min(num_moabs, count)):
                idx = count - 1 - m * 4
                hp[idx] = 200.0 * (1.0 + max(0, round_num - 40) * 0.18)
                speed[idx] = 48.0
                is_moab[idx] = 1
                children[idx] = 4
                child_hp[idx] = 10.0
                child_speed[idx] = 95.0
                rbe[idx] = hp[idx] + 40.0

        self.spawn_tick = ticks
        self.spawn_hp = hp
        self.spawn_speed = speed
        self.spawn_camo = camo
        self.spawn_immun = immun
        self.spawn_rbe = rbe
        self.spawn_children = children
        self.spawn_child_hp = child_hp
        self.spawn_child_speed = child_speed
        self.spawn_child_immun = child_immun
        self.spawn_is_moab = is_moab
        self.spawn_cursor_arr[0] = 0
        self.game_state[2] = 0.0  # Reseta tick da ronda

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.b_active.fill(0)
        self.t_active.fill(0)
        self.t_type_id.fill(-1)
        self.t_pops.fill(0.0)
        self.current_round = 1
        self.game_state[:] = [1400.0, 150.0, 0.0, 0.0]  # Cash inicial p/ build tática
        self._build_round_wave(self.current_round)
        return self._get_obs(), {}

    def action_masks(self):
        mask = np.zeros(self.total_actions, dtype=bool)
        mask[0] = True  # Esperar/Avançar é sempre válido
        cash = self.game_state[0]

        for a_idx in range(self.max_towers):
            if self.t_active[a_idx] == 0:
                for t_id, spec in enumerate(TOWER_CATALOG):
                    if cash >= spec[1]:
                        mask[1 + a_idx * self.num_tower_types + t_id] = True
            else:
                # Permite alternar modo de alvo (First <-> Strong)
                if self.t_is_village[a_idx] == 0 and self.t_income[a_idx] == 0.0:
                    mask[1 + self.max_towers * self.num_tower_types + a_idx] = True
        return mask

    def step(self, action):
        prev_rbe = self.game_state[3]
        prev_lives = self.game_state[1]
        synergy_bonus = 0.0

        if action > 0:
            build_limit = 1 + self.max_towers * self.num_tower_types
            if action < build_limit:
                rel = action - 1
                a_idx = rel // self.num_tower_types
                t_id = rel % self.num_tower_types
                spec = TOWER_CATALOG[t_id]
                if self.t_active[a_idx] == 0 and self.game_state[0] >= spec[1]:
                    self.game_state[0] -= spec[1]
                    self.t_active[a_idx] = 1
                    self.t_type_id[a_idx] = t_id
                    self.t_range[a_idx] = spec[2]
                    self.t_damage[a_idx] = spec[3]
                    self.t_moab_bonus[a_idx] = spec[4]
                    self.t_pierce[a_idx] = spec[5]
                    self.t_cooldown[a_idx] = spec[6]
                    self.t_timer[a_idx] = 0.0
                    self.t_camo_see[a_idx] = spec[7]
                    self.t_dmg_type[a_idx] = spec[8]
                    self.t_splash_radius[a_idx] = spec[9]
                    self.t_target_mode[a_idx] = TARGET_STRONG if spec[4] > 5.0 else TARGET_FIRST
                    self.t_is_alch[a_idx] = spec[10]
                    self.t_is_village[a_idx] = spec[11]
                    self.t_income[a_idx] = spec[12]

                    # Recompensa de Sinergia (Alquimista/Vila perto de DPS)
                    for j in range(self.max_towers):
                        if j != a_idx and self.t_active[j] == 1:
                            d = np.hypot(self.t_x[a_idx] - self.t_x[j], self.t_y[a_idx] - self.t_y[j])
                            if d <= 140.0 and (self.t_is_alch[a_idx] or self.t_is_village[a_idx] or self.t_is_alch[j] or self.t_is_village[j]):
                                synergy_bonus += 2.5
            else:
                a_idx = action - build_limit
                if 0 <= a_idx < self.max_towers and self.t_active[a_idx] == 1:
                    self.t_target_mode[a_idx] = (
                        TARGET_STRONG if self.t_target_mode[a_idx] == TARGET_FIRST else TARGET_FIRST
                    )

        # Avança 30 ticks de simulação de uma só vez no kernel compilado
        self.laser_count_arr[0] = 0
        step_macro_ticks_jit(
            30,
            1.0 / 60.0,
            self.wp_coords,
            self.wp_cum_dist,
            self.total_track_len,
            self.b_active,
            self.b_hp,
            self.b_max_hp,
            self.b_speed,
            self.b_dist,
            self.b_x,
            self.b_y,
            self.b_camo,
            self.b_immun,
            self.b_rbe,
            self.b_children_count,
            self.b_child_hp,
            self.b_child_speed,
            self.b_child_immun,
            self.b_is_moab,
            self.spawn_tick,
            self.spawn_hp,
            self.spawn_speed,
            self.spawn_camo,
            self.spawn_immun,
            self.spawn_rbe,
            self.spawn_children,
            self.spawn_child_hp,
            self.spawn_child_speed,
            self.spawn_child_immun,
            self.spawn_is_moab,
            self.spawn_cursor_arr,
            self.t_active,
            self.t_x,
            self.t_y,
            self.t_range,
            self.t_damage,
            self.t_moab_bonus,
            self.t_pierce,
            self.t_cooldown,
            self.t_timer,
            self.t_camo_see,
            self.t_dmg_type,
            self.t_splash_radius,
            self.t_target_mode,
            self.t_is_alch,
            self.t_is_village,
            self.t_income,
            self.t_pops,
            self.laser_buf,
            self.laser_count_arr,
            self.game_state,
        )

        rbe_delta = self.game_state[3] - prev_rbe
        lives_lost = prev_lives - self.game_state[1]
        reward = rbe_delta * 0.02 + synergy_bonus - lives_lost * 1.2

        # Verifica término da ronda
        terminated = False
        if self.game_state[1] <= 0.0:
            reward -= 50.0
            terminated = True
        elif self.spawn_cursor_arr[0] >= len(self.spawn_tick) and np.sum(self.b_active) == 0:
            # Ronda vencida: paga bônus base + renda das Farms
            farm_income = float(np.sum(self.t_income[self.t_active == 1]))
            self.game_state[0] += 100.0 + self.current_round * 12.0 + farm_income
            reward += 15.0 + farm_income * 0.01

            # Penalidade leve por acumular mais de $12k sem investir (evita hoarding passivo)
            if self.game_state[0] > 12000.0 and np.sum(self.t_active) < self.max_towers:
                reward -= 2.0

            if self.current_round >= self.max_rounds:
                reward += 250.0
                terminated = True
            else:
                self.current_round += 1
                self._build_round_wave(self.current_round)

        return self._get_obs(), float(reward), terminated, False, {}

    def _get_obs(self):
        obs = np.zeros(6 + self.max_towers * 4, dtype=np.float32)
        obs[0] = self.game_state[0] / 10000.0
        obs[1] = self.game_state[1] / 150.0
        obs[2] = self.current_round / float(self.max_rounds)
        obs[3] = np.sum(self.b_active) / 100.0
        obs[4] = np.sum(self.b_is_moab[self.b_active == 1]) / 10.0
        obs[5] = np.sum(self.t_active) / float(self.max_towers)

        base = 6
        for i in range(self.max_towers):
            obs[base + i * 4] = float(self.t_active[i])
            obs[base + i * 4 + 1] = float(self.t_type_id[i] + 1) / float(self.num_tower_types)
            obs[base + i * 4 + 2] = float(self.t_target_mode[i]) / 3.0
            obs[base + i * 4 + 3] = np.log1p(self.t_pops[i]) / 12.0
        return obs