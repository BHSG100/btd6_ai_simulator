import numpy as np
from numba import njit

# Bitmasks de Tipos de Dano e Imunidades
DMG_SHARP = 1 << 0
DMG_EXPLOSION = 1 << 1
DMG_ENERGY = 1 << 2
DMG_FREEZE = 1 << 3
DMG_NORMAL = 1 << 4  # Ignora imunidades (MIB / Acidic Mixture Dip / Tier 5)

# Modos de Alvo
TARGET_FIRST = 0
TARGET_LAST = 1
TARGET_CLOSE = 2
TARGET_STRONG = 3


@njit(fastmath=True, nogil=True, cache=True)
def step_macro_ticks_jit(
    n_ticks,
    dt,
    # Waypoints da trilha: shape (N_WP, 2) e distâncias acumuladas (N_WP,)
    wp_coords,
    wp_cum_dist,
    total_track_len,
    # Estado dos Bloons (SoA) - Capacidade fixa MAX_BLOONS
    b_active,
    b_hp,
    b_max_hp,
    b_speed,
    b_dist,
    b_x,
    b_y,
    b_camo,
    b_immun,
    b_rbe,
    b_children_count,
    b_child_hp,
    b_child_speed,
    b_child_immun,
    b_is_moab,
    # Fila de Spawn da Ronda
    spawn_tick,
    spawn_hp,
    spawn_speed,
    spawn_camo,
    spawn_immun,
    spawn_rbe,
    spawn_children,
    spawn_child_hp,
    spawn_child_speed,
    spawn_child_immun,
    spawn_is_moab,
    spawn_cursor_arr,  # array shape (1,) para mutação in-place
    # Estado das Torres (SoA) - Capacidade fixa MAX_TOWERS
    t_active,
    t_x,
    t_y,
    t_range,
    t_damage,
    t_moab_bonus,
    t_pierce,
    t_cooldown,
    t_timer,
    t_camo_See,
    t_dmg_type,
    t_splash_radius,
    t_target_mode,
    t_is_alch,
    t_is_village,
    t_income_per_round,
    t_pops,
    # Buffers visuais de feixes para o Pygame: shape (MAX_LASERS, 5) -> [x1, y1, x2, y2, dmg_type]
    laser_buf,
    laser_count_arr,
    # Estado Global da Partida: [cash, lives, current_tick, total_rbe_popped]
    game_state,
):
    max_bloons = b_active.shape[0]
    max_towers = t_active.shape[0]
    n_spawns = spawn_tick.shape[0]
    n_wp = wp_coords.shape[0]
    max_lasers = laser_buf.shape[0]

    # 1. Pré-cálculo de Auras e Buffs (Village + Alchemist)
    eff_range = np.copy(t_range)
    eff_damage = np.copy(t_damage)
    eff_cooldown = np.copy(t_cooldown)
    eff_camo = np.copy(t_camo_See)
    eff_dmg_type = np.copy(t_dmg_type)

    for i in range(max_towers):
        if t_active[i] == 0:
            continue
        if t_is_village[i] == 1:
            vr2 = t_range[i] * t_range[i]
            for j in range(max_towers):
                if i != j and t_active[j] == 1:
                    dx = t_x[i] - t_x[j]
                    dy = t_y[i] - t_y[j]
                    if dx * dx + dy * dy <= vr2:
                        eff_range[j] = t_range[j] * 1.15
                        eff_cooldown[j] = t_cooldown[j] * 0.85
                        eff_camo[j] = 1
                        if t_damage[i] > 0:  # MIB ativo
                            eff_dmg_type[j] = DMG_NORMAL
        elif t_is_alch[i] == 1:
            ar2 = t_range[i] * t_range[i]
            for j in range(max_towers):
                if i != j and t_active[j] == 1 and t_is_village[j] == 0:
                    dx = t_x[i] - t_x[j]
                    dy = t_y[i] - t_y[j]
                    if dx * dx + dy * dy <= ar2:
                        eff_damage[j] += 1.0
                        eff_range[j] *= 1.10
                        eff_cooldown[j] *= 0.90
                        eff_dmg_type[j] = DMG_NORMAL

    # Loop principal de Ticks
    for _ in range(n_ticks):
        if game_state[1] <= 0.0:
            break

        cur_tick = int(game_state[2])
        game_state[2] += 1.0

        # 2. Spawn de Bloons da Fila
        cursor = spawn_cursor_arr[0]
        while cursor < n_spawns and spawn_tick[cursor] <= cur_tick:
            # Busca slot livre no SoA
            for b in range(max_bloons):
                if b_active[b] == 0:
                    b_active[b] = 1
                    b_hp[b] = spawn_hp[cursor]
                    b_max_hp[b] = spawn_hp[cursor]
                    b_speed[b] = spawn_speed[cursor]
                    b_dist[b] = 0.0
                    b_x[b] = wp_coords[0, 0]
                    b_y[b] = wp_coords[0, 1]
                    b_camo[b] = spawn_camo[cursor]
                    b_immun[b] = spawn_immun[cursor]
                    b_rbe[b] = spawn_rbe[cursor]
                    b_children_count[b] = spawn_children[cursor]
                    b_child_hp[b] = spawn_child_hp[cursor]
                    b_child_speed[b] = spawn_child_speed[cursor]
                    b_child_immun[b] = spawn_child_immun[cursor]
                    b_is_moab[b] = spawn_is_moab[cursor]
                    break
            cursor += 1
        spawn_cursor_arr[0] = cursor

        # 3. Movimento Vetorizado e Interpolação nos Waypoints
        active_bloons_count = 0
        for b in range(max_bloons):
            if b_active[b] == 0:
                continue
            b_dist[b] += b_speed[b] * dt
            if b_dist[b] >= total_track_len:
                # Vazamento (Leak)
                game_state[1] -= b_rbe[b]
                b_active[b] = 0
                continue

            active_bloons_count += 1
            # Interpolação linear rápida na trilha
            dist = b_dist[b]
            for w in range(n_wp - 1):
                if dist >= wp_cum_dist[w] and dist <= wp_cum_dist[w + 1]:
                    seg_len = wp_cum_dist[w + 1] - wp_cum_dist[w]
                    t_ratio = (dist - wp_cum_dist[w]) / (seg_len + 1e-8)
                    b_x[b] = wp_coords[w, 0] + t_ratio * (wp_coords[w + 1, 0] - wp_coords[w, 0])
                    b_y[b] = wp_coords[w, 1] + t_ratio * (wp_coords[w + 1, 1] - wp_coords[w, 1])
                    break

        if active_bloons_count == 0 and spawn_cursor_arr[0] >= n_spawns:
            break

        # 4. Disparo das Torres com Seleção de Alvo, Imunidades e Cascata de Filhos
        for t in range(max_towers):
            if t_active[t] == 0 or t_is_village[t] == 1:
                continue

            if t_timer[t] > 0.0:
                t_timer[t] -= dt
                continue

            r2 = eff_range[t] * eff_range[t]
            tx = t_x[t]
            ty = t_y[t]
            can_camo = eff_camo[t]
            dtype = eff_dmg_type[t]
            mode = t_target_mode[t]

            best_b = -1
            best_score = -1e18

            for b in range(max_bloons):
                if b_active[b] == 0:
                    continue
                if b_camo[b] == 1 and can_camo == 0:
                    continue
                if dtype != DMG_NORMAL and (b_immun[b] & dtype) != 0:
                    continue

                dx = b_x[b] - tx
                dy = b_y[b] - ty
                d2 = dx * dx + dy * dy
                if d2 <= r2:
                    if mode == TARGET_FIRST:
                        score = b_dist[b]
                    elif mode == TARGET_LAST:
                        score = -b_dist[b]
                    elif mode == TARGET_CLOSE:
                        score = -d2
                    else:  # TARGET_STRONG
                        score = b_hp[b] * 10000.0 + b_dist[b]

                    if score > best_score:
                        best_score = score
                        best_b = b

            if best_b != -1:
                t_timer[t] = eff_cooldown[t]
                pierce_left = t_pierce[t]
                splash_r2 = t_splash_radius[t] * t_splash_radius[t]
                hit_x = b_x[best_b]
                hit_y = b_y[best_b]

                # Registra laser visual se houver espaço no buffer
                l_idx = laser_count_arr[0]
                if l_idx < max_lasers:
                    laser_buf[l_idx, 0] = tx
                    laser_buf[l_idx, 1] = ty
                    laser_buf[l_idx, 2] = hit_x
                    laser_buf[l_idx, 3] = hit_y
                    laser_buf[l_idx, 4] = float(dtype)
                    laser_count_arr[0] = l_idx + 1

                # Aplica dano ao alvo principal e vizinhos (se AoE ou Pierce > 1)
                for b in range(max_bloons):
                    if pierce_left <= 0:
                        break
                    if b_active[b] == 0:
                        continue
                    if b_camo[b] == 1 and can_camo == 0:
                        continue
                    if dtype != DMG_NORMAL and (b_immun[b] & dtype) != 0:
                        continue

                    if b != best_b:
                        if splash_r2 > 0.0:
                            dx = b_x[b] - hit_x
                            dy = b_y[b] - hit_y
                            if dx * dx + dy * dy > splash_r2:
                                continue
                        else:
                            dx = b_x[b] - tx
                            dy = b_y[b] - ty
                            if dx * dx + dy * dy > r2:
                                continue

                    dmg = eff_damage[t]
                    if b_is_moab[b] == 1:
                        dmg += t_moab_bonus[t]

                    dealt = min(b_hp[b], dmg)
                    b_hp[b] -= dmg
                    t_pops[t] += dealt
                    game_state[0] += dealt  # Cash por pop
                    game_state[3] += dealt  # Total RBE popped
                    pierce_left -= 1

                    # Destruição e Spawn de Filhos in-place
                    if b_hp[b] <= 0.0:
                        children = b_children_count[b]
                        parent_dist = b_dist[b]
                        parent_camo = b_camo[b]
                        c_hp = b_child_hp[b]
                        c_spd = b_child_speed[b]
                        c_imm = b_child_immun[b]
                        b_active[b] = 0

                        if children > 0 and c_hp > 0.0:
                            spawned = 0
                            for slot in range(max_bloons):
                                if b_active[slot] == 0:
                                    b_active[slot] = 1
                                    b_hp[slot] = c_hp
                                    b_max_hp[slot] = c_hp
                                    b_speed[slot] = c_spd
                                    b_dist[slot] = max(0.0, parent_dist - spawned * 6.0)
                                    b_x[slot] = hit_x
                                    b_y[slot] = hit_y
                                    b_camo[slot] = parent_camo
                                    b_immun[slot] = c_imm
                                    b_rbe[slot] = c_hp
                                    b_children_count[slot] = 0
                                    b_is_moab[slot] = 0
                                    spawned += 1
                                    if spawned >= children:
                                        break