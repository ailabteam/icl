# src/evaluate.py

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from qkd_environment import QKDSatelliteEnv
from baselines import run_baseline, random_policy, greedy_policy
from stable_baselines3 import PPO # Hoặc thuật toán bạn đã dùng

# --- 1. CONFIGURATION ---
# Sử dụng cùng cấu hình môi trường như lúc huấn luyện
EVAL_ENV_PARAMS = {
    'num_satellites': 10,
    'num_ground_stations': 5,
    'duration_hours': 24,
    'time_step_minutes': 5 
}

# Tìm file model mới nhất để load
model_dir = "models/"
list_of_files = glob.glob(os.path.join(model_dir, '*.zip'))
if not list_of_files:
    raise FileNotFoundError(f"Không tìm thấy file model .zip nào trong thư mục '{model_dir}'")
LATEST_MODEL_PATH = max(list_of_files, key=os.path.getctime)

NUM_EVAL_EPISODES = 5 # Chạy nhiều lần để lấy kết quả trung bình

# --- 2. EVALUATION FUNCTION ---
def evaluate_agent(model, env, num_episodes):
    all_ep_key_histories = []
    print(f"Evaluating DRL Agent for {num_episodes} episodes...")
    for _ in tqdm(range(num_episodes)):
        ep_key_history = []
        obs, info = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_key_history.append(info['total_key_generated'])
        all_ep_key_histories.append(ep_key_history)
    return np.mean(all_ep_key_histories, axis=0)

# --- 3. MAIN EVALUATION SCRIPT ---
if __name__ == '__main__':
    print("Initializing evaluation environment...")
    eval_env = QKDSatelliteEnv(**EVAL_ENV_PARAMS)

    # --- Evaluate DRL Agent ---
    print(f"Loading trained model from: {LATEST_MODEL_PATH}")
    trained_model = PPO.load(LATEST_MODEL_PATH, env=eval_env)
    drl_avg_history = evaluate_agent(trained_model, eval_env, NUM_EVAL_EPISODES)
    
    # --- Evaluate Baselines ---
    print(f"Evaluating Greedy Baseline for {NUM_EVAL_EPISODES} episodes...")
    greedy_histories = [run_baseline(eval_env, greedy_policy) for _ in tqdm(range(NUM_EVAL_EPISODES))]
    greedy_avg_history = np.mean(greedy_histories, axis=0)

    print(f"Evaluating Random Baseline for {NUM_EVAL_EPISODES} episodes...")
    random_histories = [run_baseline(eval_env, random_policy) for _ in tqdm(range(NUM_EVAL_EPISODES))]
    random_avg_history = np.mean(random_histories, axis=0)
    
    print("\n--- Evaluation Finished ---")
    print(f"Final Total Key (DRL): {drl_avg_history[-1] / 1e6:.2f} Mbit")
    print(f"Final Total Key (Greedy): {greedy_avg_history[-1] / 1e6:.2f} Mbit")
    print(f"Final Total Key (Random): {random_avg_history[-1] / 1e6:.2f} Mbit")
    
    # --- 4. PLOTTING AND SAVING FIGURE ---
    print("\nGenerating and saving the plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Chuyển đổi x-axis sang giờ
    time_steps = len(drl_avg_history)
    time_hours = np.linspace(0, EVAL_ENV_PARAMS['duration_hours'], time_steps)

    ax.plot(time_hours, drl_avg_history / 1e6, label='DRL Agent (PPO)', linewidth=2.5)
    ax.plot(time_hours, greedy_avg_history / 1e6, label='Greedy Policy', linestyle='--')
    ax.plot(time_hours, random_avg_history / 1e6, label='Random Policy', linestyle=':')
    
    ax.set_xlabel("Time (hours)", fontsize=12)
    ax.set_ylabel("Cumulative Secret Key (Mbit)", fontsize=12)
    ax.set_title("Performance Comparison of QKD Scheduling Policies", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True)
    
    # Lưu hình ảnh
    results_dir = "results/figures/"
    os.makedirs(results_dir, exist_ok=True)
    figure_path = os.path.join(results_dir, "performance_comparison.png")
    
    plt.savefig(figure_path, dpi=600, bbox_inches='tight')
    
    print(f"Plot saved to {figure_path}")
