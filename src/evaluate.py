# src/evaluate.py (UPDATED VERSION with MWBM and Runtime Measurement)

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time # <--- THÊM MỚI

from qkd_environment import QKDSatelliteEnv
from baselines import run_baseline, random_policy, mwbm_policy # <--- SỬA: Import mwbm_policy
from stable_baselines3 import PPO

# --- 1. CONFIGURATION ---
EVAL_ENV_PARAMS = {
    'num_satellites': 50,
    'num_ground_stations': 5,
    'duration_hours': 24,
    'time_step_minutes': 5
}

model_dir = "models/"
list_of_files = glob.glob(os.path.join(model_dir, '*.zip'))
if not list_of_files:
    raise FileNotFoundError(f"No model .zip file found in '{model_dir}'")
LATEST_MODEL_PATH = max(list_of_files, key=os.path.getctime)

NUM_EVAL_EPISODES = 5

# --- 2. EVALUATION FUNCTIONS (MODIFIED TO RETURN RUNTIME) ---
def evaluate_agent(model, env, num_episodes):
    all_ep_key_histories = []
    final_buffers_list = []
    all_ep_runtimes = [] # <--- THÊM MỚI
    
    print(f"Evaluating DRL Agent for {num_episodes} episodes...")
    for _ in tqdm(range(num_episodes)):
        ep_key_history = []
        total_inference_time = 0 # Thời gian inference của DRL cho 1 episode
        num_steps = 0
        
        obs, info = env.reset()
        done = False
        
        while not done:
            # Đo thời gian inference
            start_time = time.time()
            action, _states = model.predict(obs, deterministic=True)
            end_time = time.time()
            total_inference_time += (end_time - start_time)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_key_history.append(info['total_key_generated'])
            num_steps += 1
            
        all_ep_key_histories.append(ep_key_history)
        final_buffers_list.append(info['key_buffers'])
        all_ep_runtimes.append((total_inference_time / num_steps) * 1000) # Lưu runtime trung bình (ms/step)

    mean_history = np.mean(all_ep_key_histories, axis=0)
    std_history = np.std(all_ep_key_histories, axis=0)
    mean_final_buffers = np.mean(final_buffers_list, axis=0)
    std_final_buffers = np.std(final_buffers_list, axis=0)
    mean_runtime = np.mean(all_ep_runtimes) # <--- TRẢ VỀ RUNTIME
    
    return mean_history, std_history, mean_final_buffers, std_final_buffers, mean_runtime

def evaluate_baseline(env, policy_fn, num_episodes):
    all_ep_key_histories = []
    final_buffers_list = []
    all_ep_runtimes = [] # <--- THÊM MỚI
    
    print(f"Evaluating {policy_fn.__name__} for {num_episodes} episodes...")
    for _ in tqdm(range(num_episodes)):
        # run_baseline trả về (key_history, mean_runtime_ms)
        key_history, runtime_ms = run_baseline(env, policy_fn) # <--- SỬA: Nhận 2 giá trị trả về
        
        all_ep_key_histories.append(key_history)
        # Lấy buffers cuối cùng (đã được run_baseline cập nhật)
        final_buffers_list.append(env.key_buffers) 
        all_ep_runtimes.append(runtime_ms) # <--- LƯU RUNTIME
        
    mean_history = np.mean(all_ep_key_histories, axis=0)
    std_history = np.std(all_ep_key_histories, axis=0)
    mean_final_buffers = np.mean(final_buffers_list, axis=0)
    std_final_buffers = np.std(final_buffers_list, axis=0)
    mean_runtime = np.mean(all_ep_runtimes) # <--- TRẢ VỀ RUNTIME
    
    return mean_history, std_history, mean_final_buffers, std_final_buffers, mean_runtime

# --- 3. MAIN SCRIPT ---
if __name__ == '__main__':
    print("Initializing evaluation environment...")
    eval_env = QKDSatelliteEnv(**EVAL_ENV_PARAMS)
    ground_station_names = eval_env.gs_keys

    # --- Run Evaluations ---
    print(f"Loading trained model from: {LATEST_MODEL_PATH}")
    trained_model = PPO.load(LATEST_MODEL_PATH, env=eval_env)

    # DRL Agent
    drl_mean_h, drl_std_h, drl_mean_b, drl_std_b, drl_runtime = evaluate_agent(trained_model, eval_env, NUM_EVAL_EPISODES)
    
    # MWBM (Baseline mạnh)
    mwbm_mean_h, mwbm_std_h, mwbm_mean_b, mwbm_std_b, mwbm_runtime = evaluate_baseline(eval_env, mwbm_policy, NUM_EVAL_EPISODES)
    
    # Random Policy
    random_mean_h, random_std_h, random_mean_b, random_std_b, random_runtime = evaluate_baseline(eval_env, random_policy, NUM_EVAL_EPISODES)

    # --- In Kết quả Tóm tắt ---
    print("\n--- Evaluation Summary ---")
    print(f"| Policy | Final Key (Mbit) | Runtime (ms/step) |")
    print(f"|---|---|---|")
    print(f"| DRL Agent | {drl_mean_h[-1] / 1e6:.2f} | {drl_runtime:.3f} |")
    print(f"| MWBM/Optimal | {mwbm_mean_h[-1] / 1e6:.2f} | {mwbm_runtime:.3f} |")
    print(f"| Random Policy | {random_mean_h[-1] / 1e6:.2f} | {random_runtime:.3f} |")

    # --- 4. PROFESSIONAL PLOTTING ---
    results_dir = "results/figures/"
    os.makedirs(results_dir, exist_ok=True)

    # --- Plotting Style Configuration ---
    plt.style.use('seaborn-v0_8-paper') 
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "text.usetex": False, 
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
    })


    # === FIGURE 1: CUMULATIVE KEY WITH CONFIDENCE INTERVAL ===
    print("\nGenerating Figure 1: Performance Comparison...")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    time_hours = np.linspace(0, EVAL_ENV_PARAMS['duration_hours'], len(drl_mean_h))

    # Plot DRL
    ax1.plot(time_hours, drl_mean_h / 1e6, label='DRL Agent', color='royalblue', linewidth=2.5)
    ax1.fill_between(time_hours, (drl_mean_h - drl_std_h) / 1e6, (drl_mean_h + drl_std_h) / 1e6, color='royalblue', alpha=0.2)

    # Plot MWBM/Optimal Scheduler (thay thế cho Greedy cũ)
    ax1.plot(time_hours, mwbm_mean_h / 1e6, label='Optimal Instantaneous Scheduler (MWBM)', color='darkorange', linestyle='--')

    # Plot Random
    ax1.plot(time_hours, random_mean_h / 1e6, label='Random Policy', color='gray', linestyle=':')

    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Cumulative Secret Key (Mbit)")
    ax1.set_title("Comparison of Scheduling Policies")
    ax1.legend(loc='upper left')
    ax1.grid(linestyle='--', alpha=0.6)
    fig1.tight_layout()
    figure1_path_pdf = os.path.join(results_dir, "performance_comparison.pdf")
    plt.savefig(figure1_path_pdf)
    print(f"Figure 1 saved to {figure1_path_pdf}")

    # === FIGURE 2: BUFFER FAIRNESS WITH ERROR BARS AND LABELS ===
    print("\nGenerating Figure 2: Buffer Fairness...")
    fig2, ax2 = plt.subplots(figsize=(6, 4))

    x = np.arange(len(ground_station_names))
    width = 0.35

    rects1 = ax2.bar(x - width/2, drl_mean_b / 1e6, width, label='DRL Agent', yerr=drl_std_b / 1e6, capsize=5, color='royalblue', edgecolor='black')
    rects2 = ax2.bar(x + width/2, mwbm_mean_b / 1e6, width, label='Optimal Instantaneous Scheduler (MWBM)', yerr=mwbm_std_b / 1e6, capsize=5, color='darkorange', edgecolor='black')

    ax2.set_xlabel("Ground Stations")
    ax2.set_ylabel("Final Key in Buffer (Mbit)")
    ax2.set_title("Final Key Distribution and Fairness")
    ax2.set_xticks(x, ground_station_names, rotation=45, ha="right")
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    # Add labels on top of bars
    ax2.bar_label(rects1, padding=3, fmt='%.1f')
    ax2.bar_label(rects2, padding=3, fmt='%.1f')

    fig2.tight_layout()
    figure2_path_pdf = os.path.join(results_dir, "buffer_fairness.pdf")
    plt.savefig(figure2_path_pdf)
    print(f"Figure 2 saved to {figure2_path_pdf}")
