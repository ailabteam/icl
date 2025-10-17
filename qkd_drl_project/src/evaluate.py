# src/evaluate.py (ENHANCED PLOTTING VERSION)

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from qkd_environment import QKDSatelliteEnv
from baselines import run_baseline, random_policy, greedy_policy
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

# --- 2. EVALUATION FUNCTIONS (MODIFIED TO RETURN STD DEV) ---
def evaluate_agent(model, env, num_episodes):
    all_ep_key_histories = []
    final_buffers_list = []
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
        final_buffers_list.append(info['key_buffers'])

    mean_history = np.mean(all_ep_key_histories, axis=0)
    std_history = np.std(all_ep_key_histories, axis=0)
    mean_final_buffers = np.mean(final_buffers_list, axis=0)
    std_final_buffers = np.std(final_buffers_list, axis=0)
    return mean_history, std_history, mean_final_buffers, std_final_buffers

def evaluate_baseline(env, policy_fn, num_episodes):
    all_ep_key_histories = []
    final_buffers_list = []
    print(f"Evaluating {policy_fn.__name__} for {num_episodes} episodes...")
    for _ in tqdm(range(num_episodes)):
        key_history = run_baseline(env, policy_fn)
        all_ep_key_histories.append(key_history)
        final_buffers_list.append(env.key_buffers)

    mean_history = np.mean(all_ep_key_histories, axis=0)
    std_history = np.std(all_ep_key_histories, axis=0)
    mean_final_buffers = np.mean(final_buffers_list, axis=0)
    std_final_buffers = np.std(final_buffers_list, axis=0)
    return mean_history, std_history, mean_final_buffers, std_final_buffers

# --- 3. MAIN SCRIPT ---
if __name__ == '__main__':
    print("Initializing evaluation environment...")
    eval_env = QKDSatelliteEnv(**EVAL_ENV_PARAMS)
    ground_station_names = eval_env.gs_keys

    # --- Run Evaluations ---
    print(f"Loading trained model from: {LATEST_MODEL_PATH}")
    trained_model = PPO.load(LATEST_MODEL_PATH, env=eval_env)

    drl_mean_h, drl_std_h, drl_mean_b, drl_std_b = evaluate_agent(trained_model, eval_env, NUM_EVAL_EPISODES)
    greedy_mean_h, greedy_std_h, greedy_mean_b, greedy_std_b = evaluate_baseline(eval_env, greedy_policy, NUM_EVAL_EPISODES)
    random_mean_h, random_std_h, random_mean_b, random_std_b = evaluate_baseline(eval_env, random_policy, NUM_EVAL_EPISODES)

    print("\n--- Evaluation Finished (Mean Values) ---")
    print(f"Final Total Key (DRL): {drl_mean_h[-1] / 1e6:.2f} Mbit")
    print(f"Final Total Key (Greedy): {greedy_mean_h[-1] / 1e6:.2f} Mbit")
    print(f"Final Total Key (Random): {random_mean_h[-1] / 1e6:.2f} Mbit")

    # --- 4. PROFESSIONAL PLOTTING ---
    results_dir = "results/figures/"
    os.makedirs(results_dir, exist_ok=True)

    # --- Plotting Style Configuration ---
    plt.style.use('seaborn-v0_8-paper') # A style designed for scientific papers



    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"], # Dùng Times New Roman thay thế
        "text.usetex": False, # <-- THAY ĐỔI Ở ĐÂY
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
    })


    # === FIGURE 1: CUMULATIVE KEY WITH CONFIDENCE INTERVAL ===
    print("\nGenerating Figure 1: Performance Comparison...")
    fig1, ax1 = plt.subplots(figsize=(6, 4)) # Standard figure size for IEEE columns
    time_hours = np.linspace(0, EVAL_ENV_PARAMS['duration_hours'], len(drl_mean_h))

    # Plot DRL
    ax1.plot(time_hours, drl_mean_h / 1e6, label='DRL Agent (PPO)', color='royalblue', linewidth=2.5)
    ax1.fill_between(time_hours, (drl_mean_h - drl_std_h) / 1e6, (drl_mean_h + drl_std_h) / 1e6, color='royalblue', alpha=0.2)

    # Plot Greedy
    ax1.plot(time_hours, greedy_mean_h / 1e6, label='Greedy Policy', color='darkorange', linestyle='--')

    # Plot Random
    ax1.plot(time_hours, random_mean_h / 1e6, label='Random Policy', color='gray', linestyle=':')

    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Cumulative Secret Key (Mbit)")
    ax1.set_title("Comparison of Scheduling Policies")
    ax1.legend(loc='upper left')
    ax1.grid(linestyle='--', alpha=0.6)
    fig1.tight_layout()
    figure1_path_pdf = os.path.join(results_dir, "performance_comparison.pdf") # SAVE AS PDF
    plt.savefig(figure1_path_pdf)
    print(f"Figure 1 saved to {figure1_path_pdf}")

    # === FIGURE 2: BUFFER FAIRNESS WITH ERROR BARS AND LABELS ===
    print("\nGenerating Figure 2: Buffer Fairness...")
    fig2, ax2 = plt.subplots(figsize=(6, 4))

    x = np.arange(len(ground_station_names))
    width = 0.35

    rects1 = ax2.bar(x - width/2, drl_mean_b / 1e6, width, label='DRL Agent', yerr=drl_std_b / 1e6, capsize=5, color='royalblue', edgecolor='black')
    rects2 = ax2.bar(x + width/2, greedy_mean_b / 1e6, width, label='Greedy Policy', yerr=greedy_mean_b / 1e6, capsize=5, color='darkorange', edgecolor='black')

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
    figure2_path_pdf = os.path.join(results_dir, "buffer_fairness.pdf") # SAVE AS PDF
    plt.savefig(figure2_path_pdf)
    print(f"Figure 2 saved to {figure2_path_pdf}")
