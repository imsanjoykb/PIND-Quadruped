# Physics-Informed Neural Dynamics (PIND) for Quadruped Robots

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Overview

PIND is a hybrid physics-neural modeling framework that combines analytical rigid-body dynamics with learned residual corrections for high-fidelity prediction of quadruped robot behavior. The model enforces physical constraints including:

- ✅ Underactuated dynamics projection
- ✅ Unilateral contact constraints
- ✅ Friction cone admissibility
- ✅ Non-penetration conditions
- ✅ Energy conservation principles

### Key Features

- **Data Efficiency**: Achieves superior accuracy with 10-100× less training data than black-box methods
- **Physical Consistency**: Maintains momentum conservation and constraint satisfaction throughout rollouts
- **Modular Architecture**: Clean separation of nominal physics and learned residuals
- **Reproducible**: Comprehensive experiment tracking and seeded random state management

## Repository Structure

```
PIND-Quadruped/
├── pind/                          # Main package
│   ├── models/                    # Neural network architectures
│   │   ├── __init__.py
│   │   ├── pind.py               # PIND model
│   │   └── baselines.py          # Baseline models (MLP, NODE, Residual)
│   ├── utils/                     # Utility modules
│   │   ├── __init__.py
│   │   ├── mujoco_interface.py   # Physics engine interface
│   │   └── integrators.py        # Numerical integrators (RK4)
│   ├── data/                      # Data generation and loading
│   │   ├── __init__.py
│   │   └── generator.py          # Trajectory generation
│   └── configs/                   # Configuration files
│       ├── default.yaml
│       └── data_efficiency.yaml
├── scripts/                       # Executable scripts
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   └── visualize.py              # Visualization tools
├── experiments/                   # Experiment outputs
│   ├── logs/                     # Training logs
│   ├── checkpoints/              # Model checkpoints
│   └── results/                  # Evaluation results
├── tests/                         # Unit tests
│   ├── test_models.py
│   ├── test_physics.py
│   └── test_data.py
├── docs/                          # Documentation
│   ├── installation.md
│   ├── quickstart.md
│   └── experiments.md
├── assets/                        # Media and figures
│   └── architecture.png
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation
├── run_experiments.py             # Main experiment runner
├── .gitignore
├── LICENSE
├── CITATION.bib
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.8+ (optional, for GPU acceleration)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/imsanjoykb/PIND-Quadruped.git
cd PIND-Quadruped

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package with dependencies
pip install -e .
```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Verify Installation

```bash
python -c "import pind; print(pind.__version__)"
python -c "import mujoco; print(f'MuJoCo {mujoco.__version__} installed')"
```

## Quick Start

### 1. Generate Training Data

```bash
python -m pind.data.generator \
    --num-trajectories 20 \
    --trajectory-length 500 \
    --output data/go1_dataset.pt
```

### 2. Train PIND Model

```bash
python scripts/train.py \
    --config pind/configs/default.yaml \
    --data-path data/go1_dataset.pt \
    --output-dir experiments/pind_run1
```

### 3. Evaluate Model

```bash
python scripts/evaluate.py \
    --checkpoint experiments/pind_run1/best_model.pt \
    --data-path data/go1_dataset.pt \
    --horizon 50
```

### 4. Run Full Benchmark

```bash
python run_experiments.py --config pind/configs/data_efficiency.yaml
```

This will:
- Generate training data
- Train models across multiple data regimes (5%, 10%, 50%, 100%)
- Run 5 independent random seeds per regime
- Evaluate 50-step rollout performance
- Export results to `experiments/results/benchmark_results.csv`

## Usage Examples

### Training Different Models

```bash
# Train PIND (default)
python scripts/train.py --model pind --data-fraction 0.10

# Train baseline MLP
python scripts/train.py --model mlp --data-fraction 0.10

# Train Neural ODE
python scripts/train.py --model node --data-fraction 0.10

# Train unconstrained residual
python scripts/train.py --model residual --data-fraction 0.10
```

### Custom Configuration

Create a custom config file `my_config.yaml`:

```yaml
model:
  name: pind
  hidden_dim: 128
  eps_c: 0.002
  friction_coeff: 0.8

training:
  epochs: 30
  batch_size: 64
  learning_rate: 0.001
  weight_decay: 0.0001
  grad_clip: 5.0

loss_weights:
  beta_dyn: 1.0
  beta_cone: 10.0
  beta_pen: 10.0
  alpha_comp: 1.0

data:
  trajectory_length: 500
  dt: 0.02
  num_trajectories: 20
```

Then run:

```bash
python scripts/train.py --config my_config.yaml
```

### Visualization

```bash
# Plot training curves
python scripts/visualize.py --mode training --logdir experiments/pind_run1/logs

# Visualize rollout predictions
python scripts/visualize.py --mode rollout \
    --checkpoint experiments/pind_run1/best_model.pt \
    --data-path data/go1_dataset.pt

# Generate comparison plots
python scripts/visualize.py --mode comparison \
    --results experiments/results/benchmark_results.csv
```

## Reproducing Paper Results

To reproduce the results from the paper, run:

```bash
# Table II: Data Efficiency Comparison
python run_experiments.py --config pind/configs/data_efficiency.yaml

# Table III: Ablation Study
python run_experiments.py --config pind/configs/ablation.yaml

# Table IV: Long Horizon Evaluation
python run_experiments.py --config pind/configs/long_horizon.yaml
```

Results will be saved in `experiments/results/` and can be formatted for LaTeX using:

```bash
python scripts/format_results.py --input experiments/results/ --output paper_tables/
```

## Model Architecture

PIND consists of three key components:

1. **Nominal Dynamics Module**: Analytical rigid-body dynamics from MuJoCo
   - Mass matrix M(q)
   - Nonlinear bias forces h(q, v)
   - Contact Jacobians J_c(q)

2. **Actuator Residual Network**: Learns unmodeled drivetrain effects
   - Motor back-EMF
   - Transmission friction
   - Drive damping

3. **Contact Compliance Network**: Predicts foot-ground interaction forces
   - Smooth contact activation
   - Friction cone constraints
   - Non-penetration enforcement

### Composite Loss Function

```
L_total = L_data + β_dyn·L_dyn + β_cone·L_cone + β_pen·L_pen
```

Where:
- `L_data`: Supervised acceleration prediction loss
- `L_dyn`: Newton-Euler momentum residual
- `L_cone`: Friction cone violation penalty
- `L_pen`: Ground penetration and complementarity loss


## Testing

Run the test suite:

```bash
# All tests
pytest tests/

# Specific test module
pytest tests/test_models.py -v

# With coverage
pytest tests/ --cov=pind --cov-report=html
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/) and [MuJoCo](https://mujoco.org/)
- Unitree Go1 robot model adapted from [unitree_ros](https://github.com/unitreerobotics/unitree_ros)
- Inspired by recent advances in physics-informed machine learning

## Contact

For questions or issues, please:
- Open an issue on [GitHub](https://github.com/imsanjoykb/PIND-Quadruped/issues)
- Email: imsanjoykb@institution.edu

## Related Work

- [Neural ODEs](https://arxiv.org/abs/1806.07366)
- [Lagrangian Neural Networks](https://arxiv.org/abs/2003.04630)
- [Deep Lagrangian Networks](https://arxiv.org/abs/2003.01406)
- [Physics-Informed Neural Networks (PINNs)](https://www.sciencedirect.com/science/article/pii/S0021999118307125)

