# ATC Conflict Resolution

This repository contains the codebase for training, evaluating, and visualizing a Reinforcement Learning (RL) model for Air Traffic Control (ATC) conflict resolution. It uses Soft Actor-Critic (SAC) to navigate aircraft to their targets while avoiding conflicts with other aircraft and restricted airspaces. 

**Acknowledgments:** This project is built upon the [atcenv simulation environment by ramondalmau](https://github.com/ramondalmau/atcenv/tree/main).

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
To train a new SAC model and specify a custom name for the output folder, run:
```bash
python train_sac.py --timesteps 500000 --num-flights 10 --num-envs 4 --run-name "my_experiment"
```

### 2. Evaluating a Model
To evaluate a trained model (like the included baseline) and visualize the flights with PyGame:
```bash
python evaluate.py --model results/thisonLite/best_model/best_model.zip --episodes 10 --num-flights 5 --render
```

### 3. Hypothesis Testing & Sweeps
Run evaluation sweeps to test specific hypotheses. You must specify the run directory of the model you want to evaluate. The data is saved to CSV and automatically plotted:
```bash
python evaluate_hypotheses.py density-sweep --run-dir results/thisonLite --episodes 100 --workers 4 --save-csv
python evaluate_hypotheses.py airspace-sweep --run-dir results/thisonLite --episodes 100 --workers 4 --save-csv
python evaluate_hypotheses.py uncertainty-ablation --run-dir results/thisonLite --episodes 100 --workers 4 --save-csv
```

### 4. Visualization
Generate trajectory maps or plot training curves from a specific run directory:
```bash
python visualize.py trajectory --run-dir results/thisonLite --episodes 1 --num-flights 10
python visualize.py training --run-dir results/my_experiment
```

### 5. Benchmarking the MVP Baseline
To test the geometric MVP resolver instead of the RL model using parallel processing:
```bash
python bench_mvp.py --episodes 100 --num-flights 10 --workers 4
```
