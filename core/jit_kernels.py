from __future__ import annotations

from typing import Tuple
from numba import njit
import numpy as np
import numpy.typing as npt

from core.arenas import BloonArena, TowerArena


@njit(fastmath=True, nogil=True, inline="always")
def _interp_coord_scalar(
    d: float,
    waypoints: npt.NDArray[np.float32],
    cum_dists: npt.NDArray[np.float32],
    unit_dirs: npt.NDArray[np.float32],
) -> Tuple[float, float]:
    """Interpola a coordenada (x, y) de uma distância escalar `d` na trilha."""
    max_seg = waypoints.shape[0] - 2
    low = 0
    high = max_seg
    while low <= high:
        mid = (low + high) >> 1
        if cum_dists[mid + 1] <= d:
            low = mid + 1
        elif cum_dists[mid] > d:
            high = mid - 1
        else:
            low = mid
            break

    seg_idx = min(max(low, 0), max_seg)
    local_d = d - cum_dists[seg_idx]
    x = waypoints[seg_idx, 0] + unit_dirs[seg_idx, 0] * local_d
    y = waypoints[seg_idx, 1] + unit_dirs[seg_idx, 1] * local_d
    return float(x), float(y)


@njit(fastmath=True, nogil=True, cache=True)
def jit_step_movement(
    dt: float,
    bloons: BloonArena,
    lookup_speed: npt.NDArray[np.float32],
    lookup_rbe: npt.NDArray[np.int32],
    waypoints: npt.NDArray[np.float32],
    cum_dists: npt.NDArray[np.float32],
    unit_dirs: npt.NDArray[np.float32],
    total_length: float,
) -> Tuple[int, int]:
    """Avança balões 1D, detecta vazamentos e atualiza posições 2D em uma passagem."""
    leaked_rbe = 0
    active_count = 0
    capacity = bloons.active.shape[0]

    for i in range(capacity):
        if not bloons.active[i]:
            continue

        b_type = bloons.type_id[i]
        new_d = bloons.distance[i] + lookup_speed[b_type] * dt

        if new_d >= total_length:
            bloons.active[i] = False
            leaked_rbe += int(lookup_rbe[b_type])
        else:
            bloons.distance[i] = new_d
            x, y = _interp_coord_scalar(new_d, waypoints, cum_dists, unit_dirs)
            bloons.positions[i, 0] = x
            bloons.positions[i, 1] = y
            active_count += 1

    return leaked_rbe, active_count


@njit(fastmath=True, nogil=True, cache=True)
def jit_spawn_step(
    elapsed_time: float,
    cursor: int,
    spawn_times: npt.NDArray[np.float32],
    spawn_ids: npt.NDArray[np.int32],
    spawn_camo: npt.NDArray[np.bool_],
    bloons: BloonArena,
    lookup_hp: npt.NDArray[np.int32],
    lookup_speed: npt.NDArray[np.float32],
    waypoints: npt.NDArray[np.float32],
    cum_dists: npt.NDArray[np.float32],
    unit_dirs: npt.NDArray[np.float32],
    total_length: float,
) -> int:
    """Emite balões da timeline com compensação sub-tick e flag Camo."""
    total_spawns = spawn_times.shape[0]
    capacity = bloons.active.shape[0]
    free_search_start = 0

    while cursor < total_spawns and spawn_times[cursor] <= elapsed_time:
        slot = -1
        for i in range(free_search_start, capacity):
            if not bloons.active[i]:
                slot = i
                free_search_start = i + 1
                break

        if slot == -1:
            break

        b_id = spawn_ids[cursor]
        time_alive = max(0.0, elapsed_time - spawn_times[cursor])
        init_d = min(time_alive * lookup_speed[b_id], total_length)

        bloons.active[slot] = True
        bloons.type_id[slot] = b_id
        bloons.hp[slot] = lookup_hp[b_id]
        bloons.is_camo[slot] = spawn_camo[cursor]
        bloons.distance[slot] = init_d

        x, y = _interp_coord_scalar(init_d, waypoints, cum_dists, unit_dirs)
        bloons.positions[slot, 0] = x
        bloons.positions[slot, 1] = y

        cursor += 1

    return cursor


@njit(fastmath=True, nogil=True, inline="always")
def _jit_damage_slot_camo_aware(
    slot: int,
    damage: int,
    bloons: BloonArena,
    lookup_hp: npt.NDArray[np.int32],
    lookup_cash: npt.NDArray[np.int32],
    lookup_child_id: npt.NDArray[np.int32],
    lookup_child_cnt: npt.NDArray[np.int32],
    waypoints: npt.NDArray[np.float32],
    cum_dists: npt.NDArray[np.float32],
    unit_dirs: npt.NDArray[np.float32],
    work_queue: npt.NDArray[np.int32],
) -> Tuple[int, int]:
    """Aplica dano multi-camada com herança de Camo e fila estática sem alocação."""
    bloons.hp[slot] -= damage
    if bloons.hp[slot] > 0:
        return 0, 0

    cash_earned = 0
    pops = 0
    capacity = bloons.active.shape[0]
    free_hint = 0

    q_head = 0
    q_tail = 1
    work_queue[0] = slot

    while q_head < q_tail:
        curr_slot = work_queue[q_head]
        q_head += 1

        while bloons.active[curr_slot] and bloons.hp[curr_slot] <= 0:
            b_type = bloons.type_id[curr_slot]
            parent_camo = bloons.is_camo[curr_slot]
            cash_earned += int(lookup_cash[b_type])
            pops += 1

            overflow = -bloons.hp[curr_slot]
            c_id = lookup_child_id[b_type]
            c_cnt = lookup_child_cnt[b_type]

            if c_id < 0 or c_cnt <= 0:
                bloons.active[curr_slot] = False
                break

            parent_dist = bloons.distance[curr_slot]

            for extra_idx in range(1, c_cnt):
                spawn_slot = -1
                for s in range(free_hint, capacity):
                    if not bloons.active[s]:
                        spawn_slot = s
                        free_hint = s + 1
                        break
                if spawn_slot == -1:
                    for s in range(0, free_hint):
                        if not bloons.active[s]:
                            spawn_slot = s
                            free_hint = s + 1
                            break

                if spawn_slot != -1:
                    child_d = max(0.0, parent_dist - (extra_idx * 0.08))
                    bloons.active[spawn_slot] = True
                    bloons.type_id[spawn_slot] = c_id
                    bloons.hp[spawn_slot] = lookup_hp[c_id] - overflow
                    bloons.is_camo[spawn_slot] = parent_camo
                    bloons.distance[spawn_slot] = child_d
                    cx, cy = _interp_coord_scalar(child_d, waypoints, cum_dists, unit_dirs)
                    bloons.positions[spawn_slot, 0] = cx
                    bloons.positions[spawn_slot, 1] = cy

                    if bloons.hp[spawn_slot] <= 0 and q_tail < work_queue.shape[0]:
                        work_queue[q_tail] = spawn_slot
                        q_tail += 1

            bloons.type_id[curr_slot] = c_id
            bloons.hp[curr_slot] = lookup_hp[c_id] - overflow

    return cash_earned, pops


@njit(fastmath=True, nogil=True, inline="always")
def _jit_apply_alchemist_buff(t_idx: int, towers: TowerArena) -> bool:
    """Aplica poção do Alchemist na torre atacante mais próxima (priorizando sem buff)."""
    ax = towers.positions[t_idx, 0]
    ay = towers.positions[t_idx, 1]
    r_sq = towers.eff_range_sq[t_idx]
    max_towers = towers.active.shape[0]

    best_target = -1
    best_score = 1.0e30

    for other_idx in range(max_towers):
        if not towers.active[other_idx] or other_idx == t_idx:
            continue
        if towers.role[other_idx] != 0:
            continue

        dx = towers.positions[other_idx, 0] - ax
        dy = towers.positions[other_idx, 1] - ay
        d_sq = dx * dx + dy * dy

        if d_sq <= r_sq:
            is_buffed = (towers.dyn_timer[other_idx] > 0.0) and (
                towers.dyn_shots_left[other_idx] > 0
            )
            score = d_sq + (10000.0 if is_buffed else 0.0)
            if score < best_score:
                best_score = score
                best_target = other_idx

    if best_target == -1:
        return False

    towers.dyn_timer[best_target] = towers.emit_buff_dur[t_idx]
    towers.dyn_shots_left[best_target] = towers.emit_buff_shots[t_idx]
    towers.dyn_dmg_bonus[best_target] = towers.emit_buff_dmg[t_idx]
    towers.dyn_pierce_bonus[best_target] = towers.emit_buff_pierce[t_idx]
    towers.dyn_cd_mult[best_target] = towers.emit_buff_cd_mult[t_idx]
    towers.dyn_lead_bypass[best_target] = towers.emit_buff_lead[t_idx]
    return True


@njit(fastmath=True, nogil=True, cache=True)
def jit_step_combat_v2(
    dt: float,
    towers: TowerArena,
    bloons: BloonArena,
    lookup_hp: npt.NDArray[np.int32],
    lookup_rbe: npt.NDArray[np.int32],
    lookup_cash: npt.NDArray[np.int32],
    lookup_immunity: npt.NDArray[np.int32],
    lookup_child_id: npt.NDArray[np.int32],
    lookup_child_cnt: npt.NDArray[np.int32],
    waypoints: npt.NDArray[np.float32],
    cum_dists: npt.NDArray[np.float32],
    unit_dirs: npt.NDArray[np.float32],
    total_length: float,
    cand_slots: npt.NDArray[np.int32],
    hit_slots: npt.NDArray[np.int32],
    work_queue: npt.NDArray[np.int32],
) -> Tuple[int, int]:
    """Kernel de combate unificado com AoE, Camo e buffs dinâmicos."""
    total_cash = 0
    total_pops = 0
    max_towers = towers.active.shape[0]
    max_bloons = bloons.active.shape[0]
    corridor_rad_sq = 2.25

    for t_idx in range(max_towers):
        if not towers.active[t_idx]:
            continue
        if towers.timers[t_idx] > 0.0:
            towers.timers[t_idx] = max(0.0, towers.timers[t_idx] - dt)
        if towers.dyn_timer[t_idx] > 0.0:
            towers.dyn_timer[t_idx] -= dt
            if towers.dyn_timer[t_idx] <= 0.0:
                towers.dyn_shots_left[t_idx] = 0

    for t_idx in range(max_towers):
        if not towers.active[t_idx] or towers.timers[t_idx] > 0.0:
            continue

        role = towers.role[t_idx]
        if role == 1:
            continue

        if role == 2 and towers.emit_buff_shots[t_idx] > 0:
            if _jit_apply_alchemist_buff(t_idx, towers):
                towers.timers[t_idx] = towers.eff_cooldown[t_idx]
                continue

        has_dyn_buff = (towers.dyn_timer[t_idx] > 0.0) and (
            towers.dyn_shots_left[t_idx] > 0
        )

        tx = towers.positions[t_idx, 0]
        ty = towers.positions[t_idx, 1]
        r_sq = towers.eff_range_sq[t_idx]
        can_camo = towers.eff_camo[t_idx]
        mode = towers.target_mode[t_idx]

        dmg_mask = towers.eff_dmg_mask[t_idx]
        if has_dyn_buff and towers.dyn_lead_bypass[t_idx]:
            dmg_mask = dmg_mask & (~1)

        cand_count = 0
        primary_slot = -1
        best_metric = -1.0e30 if mode != 1 and mode != 2 else 1.0e30

        for b_idx in range(max_bloons):
            if not bloons.active[b_idx]:
                continue
            if bloons.is_camo[b_idx] and not can_camo:
                continue

            bt = bloons.type_id[b_idx]
            if (lookup_immunity[bt] & dmg_mask) != 0:
                continue

            dx = bloons.positions[b_idx, 0] - tx
            dy = bloons.positions[b_idx, 1] - ty
            d_sq = dx * dx + dy * dy

            if d_sq <= r_sq:
                cand_slots[cand_count] = b_idx
                cand_count += 1

                td = bloons.distance[b_idx]
                if mode == 0:
                    if td > best_metric:
                        best_metric = td
                        primary_slot = b_idx
                elif mode == 1:
                    if td < best_metric:
                        best_metric = td
                        primary_slot = b_idx
                elif mode == 2:
                    if d_sq < best_metric:
                        best_metric = d_sq
                        primary_slot = b_idx
                else:
                    score = float(lookup_rbe[bt]) * (total_length + 1.0) + td
                    if score > best_metric:
                        best_metric = score
                        primary_slot = b_idx

        if primary_slot == -1:
            continue

        dmg = int(towers.eff_damage[t_idx])
        pierce = int(towers.eff_pierce[t_idx])
        cd = float(towers.eff_cooldown[t_idx])

        if has_dyn_buff:
            dmg += int(towers.dyn_dmg_bonus[t_idx])
            pierce += int(towers.dyn_pierce_bonus[t_idx])
            cd *= float(towers.dyn_cd_mult[t_idx])
            towers.dyn_shots_left[t_idx] -= 1
            if towers.dyn_shots_left[t_idx] <= 0:
                towers.dyn_timer[t_idx] = 0.0

        blast_sq = towers.base_blast_sq[t_idx]
        impact_radius_sq = blast_sq if blast_sq > 0.0 else corridor_rad_sq

        hit_count = 1
        hit_slots[0] = primary_slot

        if pierce > 1:
            px = bloons.positions[primary_slot, 0]
            py = bloons.positions[primary_slot, 1]

            if blast_sq > 0.0:
                for b_idx in range(max_bloons):
                    if hit_count >= pierce:
                        break
                    if not bloons.active[b_idx] or b_idx == primary_slot:
                        continue
                    bt = bloons.type_id[b_idx]
                    if (lookup_immunity[bt] & dmg_mask) != 0:
                        continue
                    pdx = bloons.positions[b_idx, 0] - px
                    pdy = bloons.positions[b_idx, 1] - py
                    if (pdx * pdx + pdy * pdy) <= impact_radius_sq:
                        hit_slots[hit_count] = b_idx
                        hit_count += 1
            else:
                for c_i in range(cand_count):
                    if hit_count >= pierce:
                        break
                    s_idx = cand_slots[c_i]
                    if s_idx == primary_slot:
                        continue
                    pdx = bloons.positions[s_idx, 0] - px
                    pdy = bloons.positions[s_idx, 1] - py
                    if (pdx * pdx + pdy * pdy) <= impact_radius_sq:
                        hit_slots[hit_count] = s_idx
                        hit_count += 1

        tower_pops = 0
        for h_i in range(hit_count):
            c_earned, p_cnt = _jit_damage_slot_camo_aware(
                hit_slots[h_i],
                dmg,
                bloons,
                lookup_hp,
                lookup_cash,
                lookup_child_id,
                lookup_child_cnt,
                waypoints,
                cum_dists,
                unit_dirs,
                work_queue,
            )
            total_cash += c_earned
            total_pops += p_cnt
            tower_pops += p_cnt

        towers.timers[t_idx] = cd
        towers.pop_counter[t_idx] += tower_pops

    return total_cash, total_pops


@njit(fastmath=True, nogil=True, cache=True)
def jit_run_round_loop(
    dt: float,
    max_ticks: int,
    initial_lives: int,
    spawn_times: npt.NDArray[np.float32],
    spawn_ids: npt.NDArray[np.int32],
    spawn_camo: npt.NDArray[np.bool_],
    towers: TowerArena,
    bloons: BloonArena,
    lookup_hp: npt.NDArray[np.int32],
    lookup_speed: npt.NDArray[np.float32],
    lookup_rbe: npt.NDArray[np.int32],
    lookup_cash: npt.NDArray[np.int32],
    lookup_immunity: npt.NDArray[np.int32],
    lookup_child_id: npt.NDArray[np.int32],
    lookup_child_cnt: npt.NDArray[np.int32],
    waypoints: npt.NDArray[np.float32],
    cum_dists: npt.NDArray[np.float32],
    unit_dirs: npt.NDArray[np.float32],
    total_length: float,
    cand_slots: npt.NDArray[np.int32],
    hit_slots: npt.NDArray[np.int32],
    work_queue: npt.NDArray[np.int32],
) -> Tuple[int, int, int, int, bool]:
    """Executa uma wave inteira em código nativo LLVM sem retornar ao Python."""
    elapsed_time = 0.0
    cursor = 0
    total_spawns = spawn_times.shape[0]

    acc_cash = 0
    acc_pops = 0
    acc_lives_lost = 0
    lives = initial_lives
    ticks = 0
    round_cleared = False

    while ticks < max_ticks:
        ticks += 1

        leaked, _ = jit_step_movement(
            dt,
            bloons,
            lookup_speed,
            lookup_rbe,
            waypoints,
            cum_dists,
            unit_dirs,
            total_length,
        )
        if leaked > 0:
            acc_lives_lost += leaked
            lives -= leaked
            if lives <= 0:
                break

        elapsed_time += dt
        if cursor < total_spawns:
            cursor = jit_spawn_step(
                elapsed_time,
                cursor,
                spawn_times,
                spawn_ids,
                spawn_camo,
                bloons,
                lookup_hp,
                lookup_speed,
                waypoints,
                cum_dists,
                unit_dirs,
                total_length,
            )

        c_earned, pops = jit_step_combat_v2(
            dt,
            towers,
            bloons,
            lookup_hp,
            lookup_rbe,
            lookup_cash,
            lookup_immunity,
            lookup_child_id,
            lookup_child_cnt,
            waypoints,
            cum_dists,
            unit_dirs,
            total_length,
            cand_slots,
            hit_slots,
            work_queue,
        )
        acc_cash += c_earned
        acc_pops += pops

        if cursor >= total_spawns:
            any_alive = False
            for i in range(bloons.active.shape[0]):
                if bloons.active[i]:
                    any_alive = True
                    break
            if not any_alive:
                round_cleared = True
                break

    return acc_cash, acc_pops, acc_lives_lost, ticks, round_cleared
