# AI-Driven Dynamic Resource Management for Satellite QKD Networks

This repository contains the source code for the simulations presented in the IEEE Communications Letters paper titled: **"AI-Driven Dynamic Resource Management for Satellite Quantum Key Distribution Networks"**.

This work proposes a Deep Reinforcement Learning (DRL) framework to solve the complex problem of dynamic link scheduling in a Low Earth Orbit (LEO) satellite-based Quantum Key Distribution (QKD) network. The goal is to maximize the network-wide secret key generation while ensuring fairness among ground stations.

## Project Structure

The project is organized as follows:

```
qkd_drl_project/
├── README.md                 # This file
├── environment.yml           # Conda environment file for reproducibility
├── notebooks/                # Jupyter notebooks for experimentation and plotting
├── src/                      # Main source code directory
│   ├── __init__.py
│   ├── qkd_environment.py    # The custom Gymnasium environment for the satellite network
│   ├── utils.py              # Helper functions, physical models (orbit, SKR), and constants
│   ├── train.py              # Script to train the DRL agent
│   ├── evaluate.py           # Script to evaluate the trained agent and baselines, and generate plots
│   └── baselines.py          # Implementation of Greedy and Random baseline policies
├── models/                   # Directory to save trained model files (.zip)
└── results/                  # Directory to save simulation results and figures
    └── figures/
```

## Getting Started

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) for environment management
- An NVIDIA GPU with CUDA drivers is highly recommended for training.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ailabteam/icl.git
    cd icl/qkd_drl_project
    ```

2.  **Create and activate the Conda environment:**
    The `environment.yml` file contains all the necessary dependencies.
    ```bash
    conda env create -f environment.yml
    conda activate qkd_drl
    ```
    *Note: The `environment.yml` file is provided below. You should create this file in your repository.*

3.  **Verify the installation:**
    Run the `utils.py` script to check the Secret Key Rate (SKR) calculation model.
    ```bash
    python src/utils.py
    ```
    You should see "PASSED" for the test cases.

## Usage

### 1. Training the DRL Agent

To train a new PPO agent from scratch, run the `train.py` script. The training process will take a significant amount of time, depending on your hardware.

```bash
python src/train.py
```
- Training progress will be printed to the console.
- A trained model file (e.g., `PPO_YYYY-MM-DD_HHMMSS.zip`) will be saved in the `models/` directory.
- TensorBoard logs will be saved in the `logs/` directory.

### 2. Evaluating the Trained Agent

After training is complete, run the `evaluate.py` script to benchmark the agent against the baseline policies.

```bash
python src/evaluate.py
```
- The script will automatically load the most recently trained model from the `models/` directory.
- It will run simulations for the DRL agent, Greedy policy, and Random policy.
- Final numerical results will be printed to the console.
- Two plot files, `performance_comparison.png` and `buffer_fairness.png`, will be generated and saved in the `results/figures/` directory. These are the main figures for the paper.

## Citation

If you use this code in your research, please cite our paper:
```bibtex
@ARTICLE{Do2025ICL,
  author={Do, Phuc Hao},
  journal={IEEE Communications Letters}, 
  title={AI-Driven Dynamic Resource Management for Satellite Quantum Key Distribution Networks}, 
  year={2025},
  volume={},
  number={},
  pages={},
  doi={}
}
% Note: Please update the volume, number, pages, and DOI once the paper is published.
```

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.
