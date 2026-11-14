import pygame
import pygame.gfxdraw
import numpy as np
from ai.smart_env import TOWER_CATALOG
from core.engine import DMG_NORMAL, DMG_ENERGY

TOWER_COLORS = [
    (56, 189, 248),   # Dart Crossbow - Cyan
    (251, 146, 60),   # Bomb Recursive - Orange
    (244, 63, 94),    # Ninja Bloonjitsu - Crimson
    (192, 132, 252),  # Alchemist - Purple
    (250, 204, 21),   # Super Dark Knight - Gold
    (45, 212, 191),   # Village MIB - Teal
    (74, 222, 128),   # Farm Bank - Emerald
    (148, 163, 184),  # Sniper Maim MOAB - Steel
]


class CyberArenaVisualizer:
    """Visualizador Multi-Arena de Alta Performance com Anti-Aliasing, Heatmap e Telemetria."""

    def __init__(self, num_arenas=12, width=1600, height=920):
        pygame.init()
        pygame.display.set_caption("BTD6 AI Simulator v2.0 Pro — Cyber-Tactical Arena & Neural Telemetry")
        self.width = width
        self.height = height
        self.hud_w = 360
        self.grid_w = width - self.hud_w
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        self.cols = 4
        self.rows = 3
        self.cell_w = self.grid_w // self.cols
        self.cell_h = height // self.rows

        self.font_xs = pygame.font.SysFont("Consolas", 12)
        self.font_sm = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_md = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_lg = pygame.font.SysFont("Consolas", 22, bold=True)

        self.show_ranges = True
        self.focus_best_only = False
        self.speed_multiplier = 2
        self.paused = False
        self._bg_cache = None

    def _build_cell_bg(self, env, w, h):
        surf = pygame.Surface((w, h))
        surf.fill((13, 18, 30))
        sx, sy = w / 800.0, h / 600.0

        # Grade sutil de fundo
        for gx in range(0, w, 40):
            pygame.draw.line(surf, (20, 27, 45), (gx, 0), (gx, h), 1)
        for gy in range(0, h, 40):
            pygame.draw.line(surf, (20, 27, 45), (0, gy), (w, gy), 1)

        # Trilha com borda neon e núcleo escuro
        pts = [(int(p[0] * sx), int(p[1] * sy)) for p in env.wp_coords]
        pygame.draw.lines(surf, (38, 52, 79), False, pts, max(10, int(26 * sx)))
        pygame.draw.lines(surf, (23, 33, 53), False, pts, max(6, int(18 * sx)))

        # Slots Táticos (Anchors)
        for ax, ay in env.anchors:
            px, py = int(ax * sx), int(ay * sy)
            pygame.gfxdraw.aacircle(surf, px, py, max(3, int(6 * sx)), (45, 65, 95))
        return surf

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.show_ranges = not self.show_ranges
                elif event.key == pygame.K_TAB:
                    self.focus_best_only = not self.focus_best_only
                elif event.key == pygame.K_1:
                    self.speed_multiplier = 1
                elif event.key == pygame.K_2:
                    self.speed_multiplier = 3
                elif event.key == pygame.K_3:
                    self.speed_multiplier = 8
                elif event.key == pygame.K_4:
                    self.speed_multiplier = 20
        return True

    def render(self, envs, fitness_scores, optimizer, tps_estimate):
        self.screen.fill((9, 13, 22))
        best_idx = int(np.argmax(fitness_scores))

        if self.focus_best_only:
            self._draw_arena(envs[best_idx], best_idx, 0, 0, self.grid_w, self.height, is_best=True, fit=fitness_scores[best_idx])
        else:
            if self._bg_cache is None:
                self._bg_cache = self._build_cell_bg(envs[0], self.cell_w, self.cell_h)
            for idx, env in enumerate(envs):
                col = idx % self.cols
                row = idx // self.cols
                x0 = col * self.cell_w
                y0 = row * self.cell_h
                self._draw_arena(
                    env,
                    idx,
                    x0,
                    y0,
                    self.cell_w,
                    self.cell_h,
                    is_best=(idx == best_idx),
                    fit=fitness_scores[idx],
                    cached_bg=self._bg_cache,
                )

        self._draw_telemetry_hud(envs[best_idx], best_idx, optimizer, tps_estimate)
        pygame.display.flip()
        self.clock.tick(60)

    def _draw_arena(self, env, idx, x0, y0, w, h, is_best=False, fit=0.0, cached_bg=None):
        if cached_bg is not None:
            self.screen.blit(cached_bg, (x0, y0))
        else:
            bg = self._build_cell_bg(env, w, h)
            self.screen.blit(bg, (x0, y0))

        sx, sy = w / 800.0, h / 600.0

        # 1. Desenha Lasers de Disparo Ativos
        n_lasers = env.laser_count_arr[0]
        for l in range(n_lasers):
            lx1 = x0 + int(env.laser_buf[l, 0] * sx)
            ly1 = y0 + int(env.laser_buf[l, 1] * sy)
            lx2 = x0 + int(env.laser_buf[l, 2] * sx)
            ly2 = y0 + int(env.laser_buf[l, 3] * sy)
            dtype = int(env.laser_buf[l, 4])
            l_col = (250, 204, 21) if dtype == DMG_ENERGY else ((192, 132, 252) if dtype == DMG_NORMAL else (125, 211, 252))
            pygame.draw.aaline(self.screen, l_col, (lx1, ly1), (lx2, ly2))

        # 2. Desenha Torres e Auras
        for t in range(env.max_towers):
            if env.t_active[t] == 0:
                continue
            tx = x0 + int(env.t_x[t] * sx)
            ty = y0 + int(env.t_y[t] * sy)
            t_id = env.t_type_id[t]
            col = TOWER_COLORS[t_id % len(TOWER_COLORS)]

            if self.show_ranges and (is_best or self.focus_best_only):
                rad = int(env.t_range[t] * sx)
                if rad > 2:
                    pygame.gfxdraw.aacircle(self.screen, tx, ty, rad, (col[0], col[1], col[2], 65))

            r_body = max(5, int(11 * sx))
            pygame.gfxdraw.filled_circle(self.screen, tx, ty, r_body, col)
            pygame.gfxdraw.aacircle(self.screen, tx, ty, r_body + 2, (255, 255, 255))

        # 3. Desenha Bloons e MOABs com Barra de Vida
        for b in range(env.max_bloons):
            if env.b_active[b] == 0:
                continue
            bx = x0 + int(env.b_x[b] * sx)
            by = y0 + int(env.b_y[b] * sy)

            if env.b_is_moab[b] == 1:
                mw, mh = max(14, int(26 * sx)), max(9, int(16 * sy))
                rect = pygame.Rect(bx - mw // 2, by - mh // 2, mw, mh)
                pygame.draw.rect(self.screen, (59, 130, 246), rect, border_radius=4)
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 1, border_radius=4)
                # Barra de HP acima do MOAB
                hp_ratio = max(0.0, min(1.0, env.b_hp[b] / (env.b_max_hp[b] + 1e-5)))
                pygame.draw.rect(self.screen, (239, 68, 68), (bx - mw // 2, by - mh // 2 - 5, mw, 3))
                pygame.draw.rect(self.screen, (34, 197, 94), (bx - mw // 2, by - mh // 2 - 5, int(mw * hp_ratio), 3))
            else:
                rad = max(3, int(6.5 * sx))
                b_col = (244, 63, 94) if env.b_hp[b] <= 3 else ((234, 179, 8) if env.b_hp[b] <= 8 else (217, 119, 6))
                pygame.gfxdraw.filled_circle(self.screen, bx, by, rad, b_col)
                if env.b_camo[b] == 1:
                    pygame.gfxdraw.aacircle(self.screen, bx, by, rad + 2, (74, 222, 128))

        # 4. Moldura e Header da Arena
        border_col = (16, 185, 129) if is_best else ((239, 68, 68) if env.game_state[1] <= 0 else (38, 52, 79))
        pygame.draw.rect(self.screen, border_col, (x0, y0, w, h), 2 if is_best else 1)

        header = f"#{idx + 1} {'[ELITE]' if is_best else ''} | R:{env.current_round} | ${int(env.game_state[0])} | HP:{int(max(0, env.game_state[1]))}"
        lbl = self.font_xs.render(header, True, (241, 245, 249))
        self.screen.blit(lbl, (x0 + 8, y0 + 6))

    def _draw_telemetry_hud(self, best_env, best_idx, optimizer, tps_estimate):
        x0 = self.grid_w
        pygame.draw.rect(self.screen, (15, 23, 42), (x0, 0, self.hud_w, self.height))
        pygame.draw.line(self.screen, (51, 65, 85), (x0, 0), (x0, self.height), 2)

        y = 20
        title = self.font_lg.render("BTD6 NEURAL ENGINE", True, (56, 189, 248))
        self.screen.blit(title, (x0 + 20, y))
        y += 36

        metrics = [
            f"Geração Atual   : #{optimizer.generation}",
            f"Recorde Ronda   : Ronda {optimizer.best_round}",
            f"Melhor Fitness  : {optimizer.best_fitness:,.1f}",
            f"Velocidade Sim. : {self.speed_multiplier}x ({'PAUSADO' if self.paused else 'ATIVO'})",
            f"Throughput Total: {int(tps_estimate):,} TPS",
            f"Arena Líder     : #{best_idx + 1} (Ronda {best_env.current_round})",
        ]
        for line in metrics:
            surf = self.font_sm.render(line, True, (226, 232, 240))
            self.screen.blit(surf, (x0 + 20, y))
            y += 24

        # Gráfico de Evolução do Fitness
        y += 14
        g_title = self.font_sm.render("CURVA DE EVOLUÇÃO (FITNESS / GEN)", True, (148, 163, 184))
        self.screen.blit(g_title, (x0 + 20, y))
        y += 22
        gx, gy, gw, gh = x0 + 20, y, self.hud_w - 40, 120
        pygame.draw.rect(self.screen, (9, 13, 22), (gx, gy, gw, gh), border_radius=6)
        pygame.draw.rect(self.screen, (51, 65, 85), (gx, gy, gw, gh), 1, border_radius=6)

        hist = optimizer.fitness_history[-40:]
        if len(hist) >= 2:
            mn, mx = min(hist), max(hist)
            span = max(1.0, mx - mn)
            pts = []
            for i, val in enumerate(hist):
                px = gx + 8 + int((i / (len(hist) - 1)) * (gw - 16))
                py = gy + gh - 10 - int(((val - mn) / span) * (gh - 20))
                pts.append((px, py))
            pygame.draw.lines(self.screen, (16, 185, 129), False, pts, 2)

        # Ranking de Torres da Arena Líder (Damage Dealt / Pops)
        y += gh + 24
        t_title = self.font_sm.render("TOP TORRES ATIVAS (ARENA LÍDER)", True, (250, 204, 21))
        self.screen.blit(t_title, (x0 + 20, y))
        y += 24

        active_indices = [i for i in range(best_env.max_towers) if best_env.t_active[i] == 1]
        active_indices.sort(key=lambda i: best_env.t_pops[i], reverse=True)

        for rank, a_i in enumerate(active_indices[:10]):
            t_id = best_env.t_type_id[a_i]
            name = TOWER_CATALOG[t_id][0]
            pops = int(best_env.t_pops[a_i])
            col = TOWER_COLORS[t_id % len(TOWER_COLORS)]
            pygame.draw.circle(self.screen, col, (x0 + 28, y + 8), 5)
            txt = self.font_xs.render(f"{rank + 1}. {name:<21} | {pops:>7,} pops", True, (203, 213, 225))
            self.screen.blit(txt, (x0 + 40, y))
            y += 20

        # Atalhos no rodapé
        y = self.height - 95
        pygame.draw.line(self.screen, (51, 65, 85), (x0 + 16, y - 10), (x0 + self.hud_w - 16, y - 10), 1)
        controls = [
            "[1/2/3/4] Velocidade (1x, 3x, 8x, 20x)",
            "[TAB] Alternar Foco (Mosaico 12x / Tela Cheia)",
            "[R] Mostrar/Ocultar Raios | [ESPAÇO] Pausar",
        ]
        for c in controls:
            surf = self.font_xs.render(c, True, (148, 163, 184))
            self.screen.blit(surf, (x0 + 20, y))
            y += 20