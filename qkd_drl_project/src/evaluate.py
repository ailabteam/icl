# src/evaluate.py

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

# Find the latest model file to load
model_dir = "models/"
list_of_files = glob.glob(os.path.join(model_dir, '*.zip'))
if not list_of_files:
    raise FileNotFoundError(f"No model .zip file found in '{model_dir}'")
LATEST_MODEL_PATH = max(list_of_files, key=os.path.getctime)

NUM_EVAL_EPISODES = 5 # Run multiple times for average results

# --- 2. EVALUATION FUNCTIONS ---
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
    
    avg_history = np.mean(all_ep_key_histories, axis=0)
    avg_final_buffers = np.mean(final_buffers_list, axis=0)
    return avg_history, avg_final_buffers

def evaluate_baseline(env, policy_fn, num_episodes):
    all_ep_key_histories = []
    final_buffers_list = []
    print(f"Evaluating {policy_fn.__name__} for {num_episodes} episodes...")
    for _ in tqdm(range(num_episodes)):
        key_history = run_baseline(env, policy_fn)
        all_ep_key_histories.append(key_history)
        # Get final buffer state from the env after the run
        final_buffers_list.append(env.key_buffers)
        
    avg_history = np.mean(all_ep_key_histories, axis=0)
    avg_final_buffers = np.mean(final_buffers_list, axis=0)
    return avg_history, avg_final_buffers

# --- 3. MAIN EVALUATION SCRIPT ---
if __name__ == '__main__':
    print("Initializing evaluation environment...")
    eval_env = QKDSatelliteEnv(**EVAL_ENV_PARAMS)
    ground_station_names = eval_env.gs_keys

    # --- Evaluate DRL Agent ---
    print(f"Loading trained model from: {LATEST_MODEL_PATH}")
    trained_model = PPO.load(LATEST_MODEL_PATH, env=eval_env)
    drl_avg_history, drl_final_buffers = evaluate_agent(trained_model, eval_env, NUM_EVAL_EPISODES)
    
    # --- Evaluate Baselines ---
    greedy_avg_history, greedy_final_buffers = evaluate_baseline(eval_env, greedy_policy, NUM_EVAL_EPISODES)
    random_avg_history, random_final_buffers = evaluate_baseline(eval_env, random_policy, NUM_EVAL_EPISODES)
    
    print("\n--- Evaluation Finished ---")
    print(f"Final Total Key (DRL): {drl_avg_history[-1] / 1e6:.2f} Mbit")
    print(f"Final Total Key (Greedy): {greedy_avg_history[-1] / 1e6:.2f} Mbit")
    print(f"Final Total Key (Random): {random_avg_history[-1] / 1e6:.2f} Mbit")
    
    # --- 4. PLOTTING AND SAVING FIGURES ---
    results_dir = "results/figures/"
    os.makedirs(results_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    # === FIGURE 1: CUMULATIVE KEY ===
    print("\nGenerating and saving Figure 1: Performance Comparison...")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    time_steps = len(drl_avg_history)
    time_hours = np.linspace(0, EVAL_ENV_PARAMS['duration_hours'], time_steps)
    ax1.plot(time_hours, drl_avg_history / 1e6, label='DRL Agent (PPO)', linewidth=2.5, color='blue')
    ax1.plot(time_hours, greedy_avg_history / 1e6, label='Greedy Policy', linestyle='--', color='orange')
    ax1.plot(time_hours, random_avg_history / 1e6, label='Random Policy', linestyle=':', color='gray')
    ax1.set_xlabel("Time (hours)", fontsize=12)
    ax1.set_ylabel("Cumulative Secret Key (Mbit)", fontsize=12)
    ax1.set_title("Performance Comparison of Scheduling Policies", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True)
    fig1.tight_layout()
    figure1_path = os.path.join(results_dir, "performance_comparison.png")
    plt.savefig(figure1_path, dpi=600)
    print(f"Figure 1 saved to {figure1_path}")

    # === FIGURE 2: BUFFER FAIRNESS (BAR CHART) ===
    print("\nGenerating and saving Figure 2: Buffer Fairness...")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    df_buffers = pd.DataFrame({
        'DRL Agent': drl_final_buffers / 1e6,  # Convert to Mbit
        'Greedy Policy': greedy_final_buffers / 1e6,
    }, index=ground_station_names)
    df_buffers.plot(kind='bar', ax=ax2, color=['blue', 'orange'], width=0.8)
    ax2.set_xlabel("Ground Stations", fontsize=12)
    ax2.set_ylabel("Final Secret Key in Buffer (Mbit)", fontsize=12)
    ax2.set_title("Final Key Distribution and Fairness", fontsize=14)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', linestyle='--')
    ax2.legend(fontsize=11)
    fig2.tight_layout()
    figure2_path = os.path.join(results_dir, "buffer_fairness.png")
    plt.savefig(figure2_path, dpi=600)
    print(f"Figure 2 saved to {figure2_path}")
