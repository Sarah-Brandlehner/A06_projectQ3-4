# ATC Conflict Resolution

This repository contains the codebase for training, evaluating, and visualizing a Reinforcement Learning (RL) model for Air Traffic Control (ATC) conflict resolution. It uses Soft Actor-Critic (SAC) to navigate aircraft to their targets while avoiding conflicts with other aircraft and restricted airspaces. 

It also includes a geometric solver (MVP resolver) that acts as a baseline for comparison.

## Project Structure

- **`train_sac.py`**: The main training script to train a new SAC policy. It uses parallel environments for faster data collection.
- **`evaluate.py`**: Script to deploy and evaluate a trained policy.
- **`evaluate_hypotheses.py`**: Evaluation suite for generating experimental data (e.g., density sweeps, airspace area sweeps, uncertainty ablations).
- **`visualize.py`**: Utilities for plotting trajectories, academic evaluation graphs, and training curves.
- **`bench_mvp.py`**: A fast, headless benchmark script specifically for evaluating the MVP baseline resolver.
- **`atcenv/`**: Contains the custom ATC gym environment and wrappers (e.g., `env.py`, `multi_agent_wrapper.py`, `mvp_resolver.py`).
- **`results/thisonLite/`**: Contains the pre-trained default model and evaluation logs.

## Usage

### General Notes
- **Multicore Processing:** Many scripts (`train_sac.py`, `bench_mvp.py`, `evaluate_hypotheses.py`, `visualize.py`) utilize multiprocessing for faster execution. The number of parallel environments or workers is usually controlled by the `--num-envs` or `--workers` argument. **Set this to a maximum of the number of CPU cores available on your machine** to prevent performance degradation or system freezes.
- **Logs:** When training, progress is logged in the `results/<model_name>/eval_logs/` folder. For the default `thisonLite` baseline model, this folder is intentionally empty as the model is already fully trained. It will fill up if you start a new training run.

### 1. Training a Model
To train a new SAC model, run:
```bash
python train_sac.py --timesteps 500000 --num-flights 10 --num-envs 4
```

### 2. Evaluating a Model
To simply evaluate the default model and see basic metrics:
```bash
python evaluate.py --episodes 10 --num-flights 5
```

### 3. Hypothesis Testing & Sweeps
Run evaluation sweeps to test specific hypotheses. The data is saved to CSV and automatically plotted:
```bash
python evaluate_hypotheses.py density-sweep --episodes 100 --save-csv
python evaluate_hypotheses.py airspace-sweep --episodes 100 --save-csv
python evaluate_hypotheses.py uncertainty-ablation --episodes 100 --save-csv
```

### 4. Visualization
Generate trajectory maps or training curves:
```bash
python visualize.py trajectory --episodes 1 --num-flights 10
python visualize.py training
```

### 5. Benchmarking the MVP Baseline
To test the geometric MVP resolver instead of the RL model:
```bash
python bench_mvp.py --episodes 100 --num-flights 10
```
