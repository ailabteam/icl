# AI-Driven Dynamic Resource Management for Satellite QKD Networks

This repository contains the source code for the simulations presented in the **IEEE Communications Letters** paper titled: *"AI-Driven Dynamic Quantum Key Distribution Management in LEO Satellite Networks"* (Manuscript CL2025-2632, Revised Version).

This work proposes a Deep Reinforcement Learning (DRL) framework to solve the complex problem of dynamic link scheduling in a Low Earth Orbit (LEO) satellite-based Quantum Key Distribution (QKD) network.

## Key Contributions (Revised) 

In response to reviewer feedback, this revision includes critical technical advancements:

1.  **Superior Long-term Optimization:** The DRL agent achieves a significant improvement in total secret key volume compared to the **Optimal Instantaneous Scheduler (MWBM)**, demonstrating the effectiveness of the DRL agent's foresight in managing dynamic LEO resources.
2.  **Computational Efficiency (Runtime Analysis):** We rigorously compare the computational efficiency, showing that the DRL agent offers decision latency orders of magnitude lower than traditional optimization methods like the MWBM, making it suitable for real-time operation.
3.  **Advanced Baseline Comparison:** We replace the simple "Greedy Policy" with the **Maximum Weight Bipartite Matching (MWBM)** scheduler, satisfying the requirement for comparison against a strong graph-based optimization baseline.

## Project Structure

The project is organized as follows:

-   `environment_cuda12.yml`: Conda environment file configured for Python 3.12 and CUDA 12.x.
-   `src/`: Main source code directory
    -   `utils.py`: Contains helper functions, including the physical models for satellite orbits and the detailed **Secret Key Rate (SKR) calculation**.
    -   `qkd_environment.py`: The custom Gymnasium environment implementing the LEO QKD MDP.
    -   `baselines.py`: Implementation of `random_policy` and the **`mwbm_policy` (Optimal Instantaneous Scheduler)**, including runtime measurement logic.
    -   `train.py`: Script to train the DRL agent (PPO).
    -   `evaluate.py`: Script to evaluate the DRL agent against baselines, measure computational runtime, and generate plots.
-   `models/`: Directory to save trained PPO model files.
-   `results/`: Directory to save simulation data and figures (PDF/PNG).

## Getting Started

### Prerequisites

-   Conda for environment management.
-   An NVIDIA GPU with CUDA drivers (e.g., CUDA 12.x).

### Installation

1.  **Clone the repository and navigate to the project root:**
    ```bash
    git clone https://github.com/ailabteam/icl.git
    cd icl
    ```
2.  **Create and activate the Conda environment (based on CUDA 12.x):**
    ```bash
    conda env create -f environment_cuda12.yml
    conda activate qkd_drl
    # Install PyTorch with CUDA 12.1 support
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
3.  **Verify GPU Installation:**
    ```bash
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    ```

### Usage

1.  **Training:** (If needed, otherwise use the provided models)
    ```bash
    python src/train.py
    ```
2.  **Evaluation and Runtime Analysis:** (Uses the latest trained model)
    ```bash
    python src/evaluate.py
    ```
    This script will print the performance summary (Key Volume and Runtime) and generate updated figures (`performance_comparison.pdf` and `buffer_fairness.pdf`) in the `results/figures/` directory.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@ARTICLE{Do2025ICL,
  author={Do, Phuc Hao},
  journal={IEEE Communications Letters}, 
  title={AI-Driven Dynamic Quantum Key Distribution Management in LEO Satellite Networks}, 
  year={2025},
  volume={},
  number={},
  pages={},
  doi={},
  note={Submitted for Publication}
}
```

