import os
import sys
import json
import time
import math
import pygame
import numpy as np

from ai.train_ga import (
    DIFFICULTY_CONFIG,
    MONKEY_MEADOW_PATH,
    TOWER_COSTS,
    TOWER_RANGES,
    TOWER_BASE_DMG,
    TOWER_BASE_PIERCE,
    TOWER_COOLDOWN,
    is_valid_crosspath,
    point_to_track_dist,
)

# Paleta de Cores Visual
BG_GRASS      = (46, 105, 48)
TRACK_BORDER  = (110, 84, 54)
TRACK_FILL    = (186, 152, 107)
PANEL_BG      = (24, 28, 36)
PANEL_CARD    = (36, 42, 54)
TEXT_WHITE    = (240, 244, 248)
TEXT_GOLD     = (255, 215, 64)
TEXT_GREEN    = (80, 220, 120)
TEXT_RED      = (255, 85, 85)

TOWER_INFO = {
    0: {"name": "Dart Monkey",    "color": (165, 105, 55)},
    1: {"name": "Bomb Shooter",   "color": (55, 60, 70)},
    2: {"name": "Alchemist",      "color": (160, 70, 210)},
    3: {"name": "Monkey Village", "color": (210, 65, 65)},
    4: {"name": "Banana Farm",    "color": (230, 195, 40)},
}

BLOON_TIERS = [
    {"name": "Red",     "color": (235, 45, 45),   "hp": 1,   "speed": 1.8, "radius": 9},
    {"name": "Blue",    "color": (45, 115, 235),  "hp": 2,   "speed": 2.2, "radius": 10},
    {"name": "Green",   "color": (50, 190, 65),   "hp": 3,   "speed": 2.6, "radius": 11},
    {"name": "Yellow",  "color": (245, 215, 40),  "hp": 4,   "speed": 3.5, "radius": 12},
    {"name": "Pink",    "color": (245, 115, 175), "hp": 5,   "speed": 3.9, "radius": 12},
    {"name": "Lead",    "color": (95, 105, 115),  "hp": 12,  "speed": 1.6, "radius": 13},
    {"name": "Ceramic", "color": (195, 115, 55),  "hp": 35,  "speed": 2.8, "radius": 14},
    {"name": "MOAB",    "color": (40, 140, 255),  "hp": 240, "speed": 1.2, "radius": 22},
    {"name": "BFB",     "color": (200, 30, 30),   "hp": 750, "speed": 0.9, "radius": 27},
]


class VisualArena:
    def __init__(self, replay_path: str = "checkpoints/champion_replay.json"):
        pygame.init()
        self.width, self.height = 1280, 800
        self.map_width = 860
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("BTD6 AI Simulator - Janela Espectadora Ao Vivo")
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("Segoe UI", 22, bold=True)
        self.font_md = pygame.font.SysFont("Segoe UI", 16, bold=True)
        self.font_sm = pygame.font.SysFont("Consolas", 13)

        self.replay_path = replay_path
        self.last_mtime = 0.0
        self.auto_reload = True
        self.sim_speed = 2  # Multiplicador de velocidade inicial (1x, 2x, 4x, 8x, 16x)
        self.paused = False

        self.load_replay_or_default()

    def load_replay_or_default(self):
        if os.path.exists(self.replay_path):
            try:
                mtime = os.path.getmtime(self.replay_path)
                with open(self.replay_path, "r", encoding="utf-8") as f:
                    self.replay_data = json.load(f)
                self.last_mtime = mtime
            except Exception:
                self.replay_data = self._create_demo_data()
        else:
            self.replay_data = self._create_demo_data()

        self.reset_match()

    def _create_demo_data(self):
        """Gera uma partida inicial caso ainda não exista checkpoint salvo."""
        return {
            "source": "Modo Demonstração (Rode train_ga.py ou train_ppo_gpu.py)",
            "difficulty": "easy",
            "target_round": 40,
            "generation": 1,
            "fitness": 50161.0,
            "round_reached": 40,
            "passed_level": True,
            "actions": [
                {"action": "PLACE", "round": 1, "slot_or_type": 0, "x": 430.0, "y": 280.0, "upgrade_path": 0},
                {"action": "PLACE", "round": 2, "slot_or_type": 0, "x": 355.0, "y": 225.0, "upgrade_path": 0},
                {"action": "PLACE", "round": 4, "slot_or_type": 1, "x": 240.0, "y": 495.0, "upgrade_path": 0},
                {"action": "PLACE", "round": 7, "slot_or_type": 2, "x": 400.0, "y": 250.0, "upgrade_path": 0},
                {"action": "UPGRADE", "round": 9, "slot_or_type": 0, "x": 0.0, "y": 0.0, "upgrade_path": 0},
                {"action": "UPGRADE", "round": 11, "slot_or_type": 0, "x": 0.0, "y": 0.0, "upgrade_path": 1},
                {"action": "PLACE", "round": 14, "slot_or_type": 3, "x": 390.0, "y": 310.0, "upgrade_path": 0},
                {"action": "UPGRADE", "round": 16, "slot_or_type": 3, "x": 0.0, "y": 0.0, "upgrade_path": 0},
                {"action": "UPGRADE", "round": 19, "slot_or_type": 4, "x": 0.0, "y": 0.0, "upgrade_path": 1},
            ],
        }

    def reset_match(self):
        diff = self.replay_data.get("difficulty", "easy")
        cfg = DIFFICULTY_CONFIG.get(diff, DIFFICULTY_CONFIG["easy"])
        self.target_round = cfg["target_round"]
        self.cash = cfg["start_cash"]
        self.lives = cfg["start_lives"]
        self.cost_mult = cfg["cost_mult"]

        self.current_round = 1
        self.towers = []
        self.bloons = []
        self.lasers = []
        self.action_log = []
        self.pops_count = 0
        self.match_finished = False
        self.victory = False

        self.spawn_round_bloons(self.current_round)
        self.execute_round_actions(self.current_round)

    def execute_round_actions(self, r: int):
        for act in self.replay_data.get("actions", []):
            if int(act.get("round", -1)) != r:
                continue

            kind = act["action"]
            slot_or_type = int(act["slot_or_type"])
            gx, gy = float(act["x"]), float(act["y"])
            up_path = int(act["upgrade_path"]) % 3

            if kind == "PLACE" and len(self.towers) < 32:
                t_id = slot_or_type % 5
                cost = float(TOWER_COSTS[t_id]) * self.cost_mult
                if self.cash >= cost and point_to_track_dist(gx, gy, MONKEY_MEADOW_PATH) >= 28.0:
                    self.cash -= cost
                    self.towers.append({
                        "type": t_id,
                        "x": gx,
                        "y": gy,
                        "top": 0,
                        "mid": 0,
                        "bot": 0,
                        "invested": cost,
                        "cooldown": 0.0,
                        "buffed_alch": False,
                        "buffed_village": False,
                    })
                    t_name = TOWER_INFO[t_id]["name"]
                    self.action_log.insert(0, f"[R{r:02d}] COLOCOU {t_name} ({int(gx)},{int(gy)})")

            elif kind == "UPGRADE" and len(self.towers) > 0:
                slot = slot_or_type % len(self.towers)
                t = self.towers[slot]
                nt, nm, nb = t["top"], t["mid"], t["bot"]
                if up_path == 0:
                    nt += 1
                elif up_path == 1:
                    nm += 1
                else:
                    nb += 1
                if is_valid_crosspath(nt, nm, nb):
                    tier = max(nt, nm, nb)
                    up_cost = (180.0 * (1.65 ** tier)) * self.cost_mult
                    if self.cash >= up_cost:
                        self.cash -= up_cost
                        t["top"], t["mid"], t["bot"] = nt, nm, nb
                        t["invested"] += up_cost
                        t_name = TOWER_INFO[t["type"]]["name"]
                        self.action_log.insert(0, f"[R{r:02d}] UPGRADE {t_name} #{slot+1} -> {nt}-{nm}-{nb}")

            elif kind == "SELL" and len(self.towers) > 0:
                slot = slot_or_type % len(self.towers)
                t = self.towers.pop(slot)
                refund = t["invested"] * 0.70
                self.cash += refund
                t_name = TOWER_INFO[t["type"]]["name"]
                self.action_log.insert(0, f"[R{r:02d}] VENDEU {t_name} (+${int(refund)})")

        self.action_log = self.action_log[:14]

    def spawn_round_bloons(self, r: int):
        self.bloons.clear()
        count = min(45, 8 + int(r * 1.4))
        for i in range(count):
            max_tier = min(len(BLOON_TIERS) - 3, r // 6)
            if r >= 40 and i == count - 1:
                tier_idx = 7 if r < 60 else 8  # MOAB ou BFB no fim da onda!
            else:
                tier_idx = min(len(BLOON_TIERS) - 3, (i + r // 3) % (max_tier + 1))

            b_info = BLOON_TIERS[tier_idx]
            hp_scale = 1.0 + max(0, r - 20) * 0.08
            self.bloons.append({
                "tier": tier_idx,
                "hp": b_info["hp"] * hp_scale,
                "max_hp": b_info["hp"] * hp_scale,
                "speed": b_info["speed"],
                "wp_idx": 0,
                "x": float(MONKEY_MEADOW_PATH[0, 0]) - i * 24.0,
                "y": float(MONKEY_MEADOW_PATH[0, 1]),
                "camo": (r >= 24 and i % 6 == 0),
            })

    def step_simulation(self):
        if self.match_finished:
            return

        # 1. Atualiza Auras (Village) e Buffs (Alchemist)
        for i, t in enumerate(self.towers):
            t["buffed_village"] = False
            t["buffed_alch"] = False
            for j, other in enumerate(self.towers):
                if i == j:
                    continue
                dist = math.hypot(t["x"] - other["x"], t["y"] - other["y"])
                if other["type"] == 3 and dist <= TOWER_RANGES[3]:
                    t["buffed_village"] = True
                elif other["type"] == 2 and dist <= TOWER_RANGES[2]:
                    t["buffed_alch"] = True

        # 2. Move Bloons pelos Waypoints
        remaining_bloons = []
        for b in self.bloons:
            if b["x"] < 0:
                b["x"] += b["speed"]
                remaining_bloons.append(b)
                continue

            target_wp = MONKEY_MEADOW_PATH[b["wp_idx"] + 1]
            dx = float(target_wp[0]) - b["x"]
            dy = float(target_wp[1]) - b["y"]
            dist = math.hypot(dx, dy)

            if dist <= b["speed"]:
                b["x"], b["y"] = float(target_wp[0]), float(target_wp[1])
                b["wp_idx"] += 1
                if b["wp_idx"] >= len(MONKEY_MEADOW_PATH) - 1:
                    # Bloon escapou!
                    self.lives = max(0, self.lives - max(1, int(b["hp"])))
                    if self.lives == 0:
                        self.match_finished = True
                        self.victory = False
                    continue
            else:
                b["x"] += (dx / dist) * b["speed"]
                b["y"] += (dy / dist) * b["speed"]

            remaining_bloons.append(b)
        self.bloons = remaining_bloons

        # 3. Disparo das Torres
        for t in self.towers:
            tid = t["type"]
            if tid in (3, 4):  # Village e Banana Farm não atiram
                continue
            if t["cooldown"] > 0:
                t["cooldown"] -= 1.0
                continue

            rng = float(TOWER_RANGES[tid]) * (1.0 + 0.08 * t["bot"])
            if t["buffed_village"]:
                rng *= 1.15

            # Procura o bloon mais avançado no alcance (Targeting: FIRST)
            best_target = None
            best_progress = -1
            for b in self.bloons:
                if b["x"] < 0:
                    continue
                if b["camo"] and not (t["bot"] >= 2 or t["buffed_village"]):
                    continue
                d = math.hypot(b["x"] - t["x"], b["y"] - t["y"])
                if d <= rng and b["wp_idx"] > best_progress:
                    best_progress = b["wp_idx"]
                    best_target = b

            if best_target is not None:
                dmg = float(TOWER_BASE_DMG[tid]) + t["top"] * 1.8 + (1.5 if t["buffed_alch"] else 0.0)
                best_target["hp"] -= dmg
                self.pops_count += 1
                self.cash += 1.2
                cd_base = float(TOWER_COOLDOWN[tid]) / (1.0 + 0.22 * t["mid"] + (0.15 if t["buffed_village"] else 0.0))
                t["cooldown"] = max(4.0, cd_base * 0.35)
                self.lasers.append({
                    "x1": t["x"], "y1": t["y"],
                    "x2": best_target["x"], "y2": best_target["y"],
                    "color": TOWER_INFO[tid]["color"], "ttl": 4
                })

                # Dano em Área (AoE) para Bomb Shooter e Alchemist
                if tid in (1, 2):
                    for other_b in self.bloons:
                        if other_b is not best_target and other_b["x"] >= 0:
                            if math.hypot(other_b["x"] - best_target["x"], other_b["y"] - best_target["y"]) <= 45.0:
                                other_b["hp"] -= dmg * 0.75

        # Remove bloons estourados
        self.bloons = [b for b in self.bloons if b["hp"] > 0]

        # 4. Avança de Round quando todos os bloons da onda terminam
        if len(self.bloons) == 0 and not self.match_finished:
            # Renda de fim de round + Banana Farms
            for t in self.towers:
                if t["type"] == 4:
                    self.cash += 90.0 + t["top"] * 120.0 + t["mid"] * 95.0
            self.cash += 100.0 + self.current_round * 3.0

            if self.current_round >= self.target_round:
                self.match_finished = True
                self.victory = True
            else:
                self.current_round += 1
                self.execute_round_actions(self.current_round)
                self.spawn_round_bloons(self.current_round)

    def draw(self):
        self.screen.fill(BG_GRASS)

        # 1. Desenha a Estrada do Monkey Meadow
        pts = [(int(p[0]), int(p[1])) for p in MONKEY_MEADOW_PATH]
        pygame.draw.lines(self.screen, TRACK_BORDER, False, pts, 46)
        pygame.draw.lines(self.screen, TRACK_FILL, False, pts, 36)

        # 2. Desenha Alcances, Auras e Torres
        mouse_x, mouse_y = pygame.mouse.get_pos()
        for idx, t in enumerate(self.towers):
            tx, ty = int(t["x"]), int(t["y"])
            tid = t["type"]
            rng = int(TOWER_RANGES[tid] * (1.0 + 0.08 * t["bot"]))

            # Aura da Monkey Village ou Mouse Hover
            if tid == 3 or math.hypot(mouse_x - tx, mouse_y - ty) <= 22:
                surf = pygame.Surface((rng * 2, rng * 2), pygame.SRCALPHA)
                aura_col = (255, 220, 90, 38) if tid == 3 else (255, 255, 255, 45)
                pygame.draw.circle(surf, aura_col, (rng, rng), rng)
                pygame.draw.circle(surf, (255, 255, 255, 110), (rng, rng), rng, 1)
                self.screen.blit(surf, (tx - rng, ty - rng))

            # Brilho de Buff do Alchemist
            if t["buffed_alch"]:
                pygame.draw.circle(self.screen, (210, 90, 255), (tx, ty), 22, 2)

            # Corpo da Torre
            pygame.draw.circle(self.screen, (20, 20, 20), (tx, ty), 18)
            pygame.draw.circle(self.screen, TOWER_INFO[tid]["color"], (tx, ty), 15)

            # Tag do Crosspath (Ex: 4-2-0)
            cross_txt = f"{t['top']}-{t['mid']}-{t['bot']}"
            lbl = self.font_sm.render(cross_txt, True, TEXT_WHITE)
            self.screen.blit(lbl, (tx - lbl.get_width() // 2, ty - 28))

        # 3. Desenha Tiros / Lasers
        active_lasers = []
        for l in self.lasers:
            pygame.draw.line(self.screen, l["color"], (int(l["x1"]), int(l["y1"])), (int(l["x2"]), int(l["y2"])), 3)
            l["ttl"] -= 1
            if l["ttl"] > 0:
                active_lasers.append(l)
        self.lasers = active_lasers

        # 4. Desenha Bloons
        for b in self.bloons:
            if b["x"] < 0:
                continue
            bx, by = int(b["x"]), int(b["y"])
            b_info = BLOON_TIERS[b["tier"]]
            rad = b_info["radius"]
            pygame.draw.circle(self.screen, (20, 20, 20), (bx, by), rad + 2)
            pygame.draw.circle(self.screen, b_info["color"], (bx, by), rad)
            if b["camo"]:
                pygame.draw.circle(self.screen, (30, 80, 35), (bx, by), rad // 2)
            if b["tier"] >= 7:  # Barra de HP para MOAB / BFB
                w_bar = 36
                hp_pct = max(0.0, b["hp"] / b["max_hp"])
                pygame.draw.rect(self.screen, (40, 40, 40), (bx - w_bar // 2, by - rad - 10, w_bar, 6))
                pygame.draw.rect(self.screen, TEXT_GREEN, (bx - w_bar // 2, by - rad - 10, int(w_bar * hp_pct), 6))

        # 5. Painel Lateral de Telemetria da IA
        pygame.draw.rect(self.screen, PANEL_BG, (self.map_width, 0, self.width - self.map_width, self.height))
        px = self.map_width + 20

        title = self.font_title.render("BTD6 AI LIVE WATCHER", True, TEXT_GOLD)
        self.screen.blit(title, (px, 18))

        diff_str = self.replay_data.get("difficulty", "easy").upper()
        gen_num = self.replay_data.get("generation", 1)
        fit_num = self.replay_data.get("fitness", 0.0)

        stats = [
            f"Dificuldade: {diff_str} (Alvo: Round {self.target_round})",
            f"Geração / Ep: #{gen_num}  |  Fitness: {fit_num:,.1f}",
            f"Round Atual: {self.current_round:02d} / {self.target_round:02d}",
            f"Vidas: {self.lives}   |   Dinheiro: ${int(self.cash):,}",
            f"Torres no Mapa: {len(self.towers)}   |   Pops: {self.pops_count:,}",
            f"Velocidade: {self.sim_speed}x ({'PAUSADO' if self.paused else 'RODANDO'})",
            f"Auto-Reload Campeão: {'ATIVO (A)' if self.auto_reload else 'DESATIVADO (A)'}",
        ]

        y_cursor = 60
        for line in stats:
            surf = self.font_md.render(line, True, TEXT_WHITE)
            self.screen.blit(surf, (px, y_cursor))
            y_cursor += 28

        # Banner de Vitória ou Derrota
        if self.match_finished:
            banner_col = TEXT_GREEN if self.victory else TEXT_RED
            msg = f"NÍVEL CONCLUÍDO! VENCEU O MODO {diff_str}!" if self.victory else f"FIM DE JOGO NO ROUND {self.current_round}!"
            b_surf = self.font_md.render(msg, True, banner_col)
            self.screen.blit(b_surf, (px, y_cursor + 6))
        y_cursor += 42

        # Log de Decisões da IA
        log_header = self.font_md.render("Últimas Jogadas da IA (Cromossomo):", True, TEXT_GOLD)
        self.screen.blit(log_header, (px, y_cursor))
        y_cursor += 26

        for entry in self.action_log:
            l_surf = self.font_sm.render(entry, True, (200, 210, 225))
            self.screen.blit(l_surf, (px, y_cursor))
            y_cursor += 20

        # Rodapé de Atalhos
        controls = [
            "[SETA DIR/ESQ] Velocidade (1x a 32x)",
            "[ESPAÇO] Pausar / Continuar",
            "[R] Reiniciar Partida Atual",
            "[A] Alternar Auto-Reload de Treino",
        ]
        cy = self.height - 95
        for c_line in controls:
            c_surf = self.font_sm.render(c_line, True, (150, 165, 185))
            self.screen.blit(c_surf, (px, cy))
            cy += 20

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_RIGHT:
                        self.sim_speed = min(32, self.sim_speed * 2)
                    elif event.key == pygame.K_LEFT:
                        self.sim_speed = max(1, self.sim_speed // 2)
                    elif event.key == pygame.K_r:
                        self.reset_match()
                    elif event.key == pygame.K_a:
                        self.auto_reload = not self.auto_reload

            # Se um novo campeão foi salvo pelo treino de madrugada, carrega automaticamente ao terminar a partida
            if self.auto_reload and os.path.exists(self.replay_path):
                mtime = os.path.getmtime(self.replay_path)
                if mtime > self.last_mtime and self.match_finished:
                    self.load_replay_or_default()

            if not self.paused:
                for _ in range(self.sim_speed):
                    self.step_simulation()

            self.draw()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    arena = VisualArena()
    arena.run()