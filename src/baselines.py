# src/baselines.py

import numpy as np
from skyfield.api import load
import time # Thêm mới

from qkd_environment import QKDSatelliteEnv
from utils import SIM_PARAMS, calculate_skr

def run_baseline(env, policy_fn):
    """
    Hàm khung để chạy một episode với một policy cho trước,
    đồng thời đo thời gian tính toán Policy (Policy Inference Time).
    """
    obs, info = env.reset()
    done = False
    total_key = 0
    key_history = []
    
    total_runtime = 0
    num_steps = 0 # Đếm bước thực tế

    while not done:
        # Đo thời gian tính toán action
        start_time = time.time()
        action = policy_fn(env, obs)
        end_time = time.time()
        
        total_runtime += (end_time - start_time)
        num_steps += 1

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_key = info['total_key_generated']
        key_history.append(total_key)

    mean_runtime_ms = (total_runtime / num_steps) * 1000 if num_steps > 0 else 0
    
    return key_history, mean_runtime_ms

def random_policy(env, obs):
    """Chọn một hành động hoàn toàn ngẫu nhiên."""
    return env.action_space.sample()

def mwbm_policy(env, obs):
    """
    Optimal Instantaneous Scheduler (MWBM) - Tối ưu hóa Key Rate tức thời.
    Trong mô hình 1-link/step, MWBM tương đương việc chọn link có SKR cao nhất.
    (Đổi tên từ Greedy Policy cũ để tăng tính học thuật)
    """
    current_time = env.start_time + env.current_step * env.time_step_seconds / 86400.0

    best_action = env.num_satellites * env.num_ground_stations # "no action"
    max_skr = -1.0

    # Duyệt qua tất cả các hành động kết nối khả thi
    # (Đoạn code này giống hệt logic Greedy cũ của bạn)
    for action in range(env.num_satellites * env.num_ground_stations):
        s_idx = action // env.num_ground_stations
        g_idx = action % env.num_ground_stations

        sat = env.satellites[s_idx]
        gs = env.ground_stations[env.gs_keys[g_idx]]

        difference = sat - gs
        topocentric = difference.at(current_time)
        alt, _, distance = topocentric.altaz()

        if alt.degrees > SIM_PARAMS['min_elevation']:
            zenith_rad = np.deg2rad(90.0 - alt.degrees)
            skr = calculate_skr(distance.km, zenith_rad)

            if skr > max_skr:
                max_skr = skr
                best_action = action

    return best_action
