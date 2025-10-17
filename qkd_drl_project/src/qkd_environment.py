# src/qkd_environment.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from skyfield.api import load

from utils import create_constellation, create_ground_stations, calculate_skr, SIM_PARAMS

class QKDSatelliteEnv(gym.Env):
    """
    Môi trường Reinforcement Learning tùy chỉnh cho bài toán quản lý QKD.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, num_satellites=5, num_ground_stations=3, duration_hours=24, time_step_minutes=1):
        super(QKDSatelliteEnv, self).__init__()

        # --- Parameters ---
        self.num_satellites = num_satellites
        self.num_ground_stations = num_ground_stations
        self.time_step_seconds = time_step_minutes * 60
        self.max_steps = int((duration_hours * 3600) / self.time_step_seconds)
        self.current_step = 0

        # --- Skyfield setup ---
        self.ts = load.timescale()
        self.start_time = self.ts.now()
        self.satellites = create_constellation(self.ts, self.num_satellites)
        
        # Tạo các trạm mặt đất ở các vị trí chiến lược
        # (Ví dụ: Hà Nội, Paris, New York)
        gs_locations = {
            'Hanoi': (21.02, 105.85, 20),
            'Paris': (48.85, 2.35, 35),
            'NewYork': (40.71, -74.00, 10)
        }
        # Chỉ lấy số lượng trạm cần thiết
        self.gs_keys = list(gs_locations.keys())[:self.num_ground_stations]
        self.ground_stations = create_ground_stations(self.ts, {k: gs_locations[k] for k in self.gs_keys})

        # --- State and Action Space Definition ---
        # Action: Chọn 1 cặp (vệ tinh, trạm) để kết nối, hoặc không làm gì.
        # Action space size = (num_satellites * num_ground_stations) + 1 (for "no action")
        self.action_space = spaces.Discrete(self.num_satellites * self.num_ground_stations + 1)

        # Observation:
        # 1. Ma trận kết nối khả dụng (S x G): 1 nếu kết nối được, 0 nếu không.
        # 2. Lượng khóa trong buffer của mỗi trạm (G values), đã được chuẩn hóa.
        # 3. Thời gian trong ngày (1 value), đã được chuẩn hóa.
        obs_size = (self.num_satellites * self.num_ground_stations) + self.num_ground_stations + 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_size,), dtype=np.float32)

        # --- Environment State Variables ---
        self.key_buffers = np.zeros(self.num_ground_stations, dtype=np.float32)
        self.max_buffer_capacity = 1e9 # 1 Gbit

    def _get_obs(self):
        """Tính toán và trả về observation hiện tại."""
        # 1. Tính ma trận kết nối khả dụng
        visibility_matrix = np.zeros((self.num_satellites, self.num_ground_stations), dtype=np.float32)
        
        current_time = self.start_time + self.current_step * self.time_step_seconds / 86400.0
        
        for s_idx, sat in enumerate(self.satellites):
            for g_idx, gs_key in enumerate(self.gs_keys):
                gs = self.ground_stations[gs_key]
                difference = sat - gs
                topocentric = difference.at(current_time)
                alt, _, _ = topocentric.altaz()
                
                if alt.degrees > SIM_PARAMS['min_elevation']:
                    visibility_matrix[s_idx, g_idx] = 1.0

        # 2. Chuẩn hóa key buffers
        normalized_buffers = self.key_buffers / self.max_buffer_capacity
        
        # 3. Chuẩn hóa thời gian
        normalized_time = (self.current_step % (24 * 60 / (self.time_step_seconds/60))) / (24 * 60 / (self.time_step_seconds/60))


        # Nối tất cả lại thành một vector
        obs = np.concatenate([
            visibility_matrix.flatten(),
            normalized_buffers,
            np.array([normalized_time])
        ])
        return obs.astype(np.float32)
        
    def _get_info(self):
        """Trả về thông tin phụ, hữu ích cho việc debug và phân tích."""
        return {
            "total_key_generated": np.sum(self.key_buffers),
            "key_buffers": self.key_buffers.copy()
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.key_buffers = np.zeros(self.num_ground_stations, dtype=np.float32)
        self.start_time = self.ts.now() # Lấy thời gian mới cho mỗi episode
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        self.current_step += 1
        current_time = self.start_time + self.current_step * self.time_step_seconds / 86400.0

        # --- Execute Action ---
        total_generated_key_this_step = 0
        
        # Decode action
        if action < self.num_satellites * self.num_ground_stations:
            s_idx = action // self.num_ground_stations
            g_idx = action % self.num_ground_stations
            
            sat = self.satellites[s_idx]
            gs = self.ground_stations[self.gs_keys[g_idx]]
            
            # Tính toán SKR cho cặp đã chọn
            difference = sat - gs
            topocentric = difference.at(current_time)
            alt, _, distance = topocentric.altaz()
            
            if alt.degrees > SIM_PARAMS['min_elevation']:
                zenith_rad = np.deg2rad(90.0 - alt.degrees)
                skr = calculate_skr(distance.km, zenith_rad)
                generated_key = skr * self.time_step_seconds
                
                # Cập nhật buffer, không để tràn
                self.key_buffers[g_idx] = min(self.max_buffer_capacity, self.key_buffers[g_idx] + generated_key)
                total_generated_key_this_step = generated_key

        # --- Calculate Reward ---
        # Phần thưởng chính là lượng khóa sinh ra
        reward = total_generated_key_this_step / 1e6 # Chuẩn hóa reward (ví dụ: Mbits)
        
        # Thêm một thành phần phạt nhỏ cho sự mất cân bằng
        if np.sum(self.key_buffers) > 0:
            fairness_penalty = np.std(self.key_buffers) / np.mean(self.key_buffers)
            reward -= fairness_penalty * 0.1 # Trọng số của hình phạt

        # --- Check for Termination ---
        terminated = self.current_step >= self.max_steps
        truncated = False # Không dùng truncated trong trường hợp này
        
        # --- Get Next Observation and Info ---
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

# ==============================================================================
# MAIN BLOCK FOR TESTING
# ==============================================================================
if __name__ == '__main__':
    # Đoạn code này để kiểm tra môi trường có tuân thủ API của Gymnasium không
    from stable_baselines3.common.env_checker import check_env
    
    print("Khởi tạo môi trường QKD Satellite...")
    env = QKDSatelliteEnv()
    
    print("Kiểm tra sự tuân thủ của môi trường với API của Gymnasium...")
    # check_env sẽ báo lỗi nếu có bất kỳ vấn đề nào.
    check_env(env)
    
    print("\nKiểm tra thành công! Môi trường đã sẵn sàng.")
    
    print("\n--- Thông tin môi trường ---")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space Shape: {env.observation_space.shape}")
    
    # Thử chạy một episode với hành động ngẫu nhiên
    print("\n--- Chạy thử 1 episode với hành động ngẫu nhiên ---")
    obs, info = env.reset()
    done = False
    step_count = 0
    total_reward = 0
    
    while not done:
        action = env.action_space.sample() # Chọn một hành động ngẫu nhiên
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1
        
        if step_count % 100 == 0:
            print(f"Step {step_count}: Total Reward={total_reward:.2f}, Total Key={info['total_key_generated']/1e6:.2f} Mbit")

    print("\nEpisode hoàn thành!")
