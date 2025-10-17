# src/baselines.py

import numpy as np
from skyfield.api import load

from qkd_environment import QKDSatelliteEnv
from utils import SIM_PARAMS, calculate_skr

def run_baseline(env, policy_fn):
    """Hàm khung để chạy một episode với một policy cho trước."""
    obs, info = env.reset()
    done = False
    total_key = 0
    key_history = []
    
    while not done:
        action = policy_fn(env, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_key = info['total_key_generated']
        key_history.append(total_key)
        
    return key_history

def random_policy(env, obs):
    """Chọn một hành động hoàn toàn ngẫu nhiên."""
    return env.action_space.sample()

def greedy_policy(env, obs):
    """
    Chọn hành động mang lại SKR tức thời cao nhất.
    Đây là một chính sách tham lam (myopic).
    """
    current_time = env.start_time + env.current_step * env.time_step_seconds / 86400.0
    
    best_action = env.num_satellites * env.num_ground_stations # "no action"
    max_skr = -1.0
    
    # --- BIẾN CỜ ĐỂ DEBUG ---
    found_positive_skr_in_step = False

    # Duyệt qua tất cả các hành động kết nối khả thi
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
            
            # --- THÊM ĐOẠN DEBUG ---
            if skr > 0:
                print(f"[DEBUG-Greedy] Step {env.current_step:03d}: Found link (Sat {s_idx:02d}, GS {g_idx}) with SKR = {skr / 1e3:.2f} kbps")
                found_positive_skr_in_step = True
            # --- KẾT THÚC DEBUG ---
            
            if skr > max_skr:
                max_skr = skr
                best_action = action
    
    # --- THÊM DEBUG ---
    # In ra kết luận cho mỗi bước nếu tìm thấy ít nhất một SKR > 0
    if found_positive_skr_in_step:
        chosen_s_idx = best_action // env.num_ground_stations
        chosen_g_idx = best_action % env.num_ground_stations
        print(f"    -> At Step {env.current_step:03d}, Greedy chose link (Sat {chosen_s_idx:02d}, GS {chosen_g_idx}) with max SKR = {max_skr / 1e3:.2f} kbps")
    # --- KẾT THÚC DEBUG ---
            
    return best_action
