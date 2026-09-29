# Quick Start Guide

## Installation

### Option 1: Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PIND-Quadruped.git
cd PIND-Quadruped

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

### Option 2: Docker

```bash
# Build and run with Docker
docker-compose up -d pind
docker exec -it pind-research bash

# Or with GPU support
docker-compose up -d pind
```

## Running Your First Experiment

### 1. Generate Data

```bash
python -m pind.data.generator \
    --num-trajectories 20 \
    --trajectory-length 500 \
    --output data/go1_dataset.pt
```

Expected output:
- Dataset file: `data/go1_dataset.pt`
- Size: ~150-200 MB
- Duration: ~2-3 minutes

### 2. Train PIND Model

```bash
python scripts/train.py \
    --config pind/configs/default.yaml \
    --data-path data/go1_dataset.pt \
    --output-dir experiments/my_first_run
```

Expected duration: ~30-45 minutes (GPU) or ~2-3 hours (CPU)

### 3. Evaluate Model

```bash
python scripts/evaluate.py \
    --checkpoint experiments/my_first_run/best_model.pt \
    --data-path data/go1_dataset.pt \
    --horizon 50
```

Expected metrics:
- Position RMSE: 2-5 mm (100% data)
- Cone Violation: < 0.05 N

### 4. Run Full Benchmark

```bash
python run_experiments.py --config pind/configs/data_efficiency.yaml
```

This will:
- Train across 4 data regimes (5%, 10%, 50%, 100%)
- Run 5 random seeds per regime
- Generate results table in `experiments/results/`

Expected duration: ~4-6 hours (GPU) or ~1-2 days (CPU)

## Understanding the Results

### Output Files

```
experiments/
├── my_first_run/
│   ├── best_model.pt          # Best model checkpoint
│   ├── config.yaml            # Experiment configuration
│   ├── logs/                  # TensorBoard logs
│   └── checkpoint_epoch_*.pt  # Periodic checkpoints
└── results/
    └── benchmark_results.csv  # Final metrics table
```

### Metrics Explained

- **Position RMSE (mm)**: Average position prediction error over 50-step rollout
- **Velocity RMSE**: Velocity prediction error
- **Cone Violation (N)**: Friction cone constraint violation magnitude
- **Penetration (mm)**: Ground penetration depth (should be near zero)

## Visualization

```bash
# Plot training curves
python scripts/visualize.py --mode training \
    --logdir experiments/my_first_run/logs

# Compare with baselines
python scripts/visualize.py --mode comparison \
    --results experiments/results/benchmark_results.csv
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in config:
```yaml
training:
  batch_size: 32  # Default is 64
```

### MuJoCo Installation Issues

```bash
# Verify MuJoCo installation
python -c "import mujoco; print(mujoco.__version__)"

# If issues persist, try:
pip install --upgrade mujoco
```

### Slow Training

- Use GPU if available
- Reduce `num_trajectories` for quick tests
- Use `--data-fraction 0.1` for 10% data experiments

## Next Steps

1. **Modify the model**: Edit `pind/models/pind.py`
2. **Custom configurations**: Create your own config in `pind/configs/`
3. **Add new baselines**: Extend `pind/models/baselines.py`
4. **Experiment tracking**: Enable Weights & Biases in config

## Getting Help

- Documentation: `docs/`
- Issues: [GitHub Issues](https://github.com/yourusername/PIND-Quadruped/issues)
- Examples: `examples/` directory

## Common Commands Cheat Sheet

```bash
# Quick test run (10% data, 5 epochs)
python scripts/train.py --data-fraction 0.1 --epochs 5

# Train different models
python scripts/train.py --model pind
python scripts/train.py --model mlp
python scripts/train.py --model node

# Evaluate with different horizons
python scripts/evaluate.py --checkpoint path/to/model.pt --horizon 100

# Generate more data
python -m pind.data.generator --num-trajectories 50

# Run tests
pytest tests/ -v
```
