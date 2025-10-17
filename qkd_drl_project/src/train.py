# src/train.py

import os
from datetime import datetime
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Sửa lại import để phù hợp với cách chạy file
from qkd_environment import QKDSatelliteEnv

# --- 1. CONFIGURATION ---
# Các tham số này có thể được đưa ra file config sau này
TRAINING_STEPS = 100_000
MODEL_ALGORITHM = PPO
ENV_PARAMS = {
    'num_satellites': 10,
    'num_ground_stations': 5,
    'duration_hours': 24,
    'time_step_minutes': 5 
}

# --- 2. SETUP DIRECTORIES AND FILENAMES ---
# Tạo thư mục để lưu logs và models
log_dir = "logs/"
model_dir = "models/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Tạo tên file duy nhất dựa trên thời gian
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
model_name = f"{MODEL_ALGORITHM.__name__}_{timestamp}"
model_save_path = os.path.join(model_dir, model_name)


# --- 3. ENVIRONMENT AND MODEL SETUP ---
print("Initializing the QKD Satellite environment...")
env = QKDSatelliteEnv(**ENV_PARAMS)

# Callback để lưu model định kỳ
checkpoint_callback = CheckpointCallback(
    save_freq=10_000, 
    save_path=model_dir, 
    name_prefix=model_name
)

# Thiết lập model PPO
# 'MlpPolicy' là một mạng neural đa lớp tiêu chuẩn
# verbose=1 để in ra thông tin huấn luyện
# device='auto' sẽ tự động chọn GPU nếu có
model = MODEL_ALGORITHM(
    'MlpPolicy', 
    env, 
    verbose=1,
    tensorboard_log=log_dir,
    device='auto' # Quan trọng: để sử dụng RTX 4090 của bạn
)

# --- 4. TRAINING ---
print(f"Starting training for {TRAINING_STEPS} steps...")
print(f"Algorithm: {MODEL_ALGORITHM.__name__}")
print(f"Using device: {model.device}")
print(f"Models will be saved with prefix: {model_name}")
print(f"TensorBoard logs available at: ./{log_dir}")
print("-" * 50)

# Bắt đầu huấn luyện!
model.learn(
    total_timesteps=TRAINING_STEPS, 
    callback=checkpoint_callback,
    tb_log_name=model_name
)

# --- 5. SAVE THE FINAL MODEL ---
print("-" * 50)
print("Training finished.")
model.save(model_save_path)
print(f"Final model saved to {model_save_path}.zip")

# Đóng môi trường
env.close()
