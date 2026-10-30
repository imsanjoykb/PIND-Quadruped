# API Documentation

Complete API reference for the PIND package.

## 📚 Table of Contents

- [Models](#models)
  - [PIND](#pind-model)
  - [Baselines](#baseline-models)
- [Utilities](#utilities)
  - [MuJoCo Interface](#mujoco-interface)
  - [Integrators](#integrators)
- [Data](#data)
  - [Data Generation](#data-generation)
  - [Dataset Class](#dataset-class)
- [Training](#training)
- [Evaluation](#evaluation)

---

## Models

### PIND Model

Main physics-informed neural dynamics model.

#### Class: `pind.models.PIND`

```python
from pind.models import PIND

model = PIND(
    nv=18,              # Number of velocity DOFs
    nu=12,              # Number of actuated joints
    hidden_dim=128,     # Hidden layer dimension
    eps_c=0.002,        # Contact activation threshold
    mu=0.8              # Friction coefficient
)
```

**Parameters:**
- `nv` (int): Velocity degrees of freedom (default: 18)
- `nu` (int): Number of actuated joints (default: 12)
- `hidden_dim` (int): Hidden layer dimension for networks (default: 128)
- `eps_c` (float): Contact activation threshold (default: 0.002)
- `mu` (float): Friction coefficient (default: 0.8)

**Methods:**

#### `forward_acceleration(q, v, u, M_nom, h_nom, p_feet, J_feet)`

Compute generalized acceleration using physics-informed dynamics.

```python
v_dot, lambdas, delta_tau = model.forward_acceleration(
    q=q,            # Positions (batch, 19)
    v=v,            # Velocities (batch, 18)
    u=u,            # Controls (batch, 12)
    M_nom=M_nom,    # Mass matrix (batch, 18, 18)
    h_nom=h_nom,    # Bias forces (batch, 18)
    p_feet=p_feet,  # Foot positions (batch, 4, 3)
    J_feet=J_feet   # Foot Jacobians (batch, 4, 3, 18)
)
```

**Returns:**
- `v_dot` (Tensor): Predicted accelerations (batch, 18)
- `lambdas` (List[Tensor]): Contact forces for each foot (4 tensors of shape (batch, 3))
- `delta_tau` (Tensor): Actuator residuals (batch, 12)

#### `compute_composite_loss(...)`

Compute physics-informed composite loss.

```python
loss, metrics = model.compute_composite_loss(
    v_dot_pred=v_dot_pred,          # Predicted accelerations
    v_dot_target=v_dot_target,      # Ground truth accelerations
    predicted_lambdas=lambdas,       # Predicted contact forces
    p_feet=p_feet,                   # Foot positions
    M_nom=M_nom,                     # Mass matrix
    h_nom=h_nom,                     # Bias forces
    u=u,                             # Control inputs
    delta_tau=delta_tau,             # Actuator residuals
    J_feet=J_feet,                   # Foot Jacobians
    beta_dyn=1.0,                    # Dynamics loss weight
    beta_cone=10.0,                  # Friction cone loss weight
    beta_pen=10.0                    # Penetration loss weight
)
```

**Returns:**
- `loss` (Tensor): Total weighted loss
- `metrics` (dict): Individual loss components
  - `l_data`: Supervised acceleration loss
  - `l_dyn`: Dynamics residual loss
  - `l_cone`: Friction cone violation loss
  - `l_pen`: Penetration/complementarity loss

---

### Baseline Models

#### Class: `pind.models.BlackBoxMLP`

Pure data-driven discrete dynamics model.

```python
from pind.models import BlackBoxMLP

model = BlackBoxMLP(
    state_dim=37,    # State dimension (q + v)
    act_dim=12,      # Action dimension
    hidden_dim=256   # Hidden layer dimension
)

# Forward pass
delta_x = model(x, u)  # x: (batch, 37), u: (batch, 12)
```

#### Class: `pind.models.NeuralODEVectorField`

Continuous-time neural ODE model.

```python
from pind.models import NeuralODEVectorField

model = NeuralODEVectorField(
    state_dim=37,
    act_dim=12,
    hidden_dim=256
)

# Forward pass
x_dot = model(x, u)  # Compute state derivative
```

#### Class: `pind.models.UnconstrainedResidualMLP`

Hybrid model without physics constraints.

```python
from pind.models import UnconstrainedResidualMLP

model = UnconstrainedResidualMLP(
    state_dim=37,
    act_dim=12,
    nv=18,
    hidden_dim=128
)

# Forward pass
v_dot = model(x, u, v_dot_nom)  # Nominal + residual
```

---

## Utilities

### MuJoCo Interface

#### Class: `pind.utils.QuadrupedMuJoCoInterface`

Interface for extracting analytical dynamics from MuJoCo.

```python
from pind.utils import QuadrupedMuJoCoInterface

# Initialize with XML path
sim = QuadrupedMuJoCoInterface(xml_path="path/to/robot.xml")

# Or use procedural model
sim = QuadrupedMuJoCoInterface()

# Compute nominal dynamics
M, h, p_feet, J_feet = sim.compute_nominal_dynamics(q, v)
```

**Parameters:**
- `xml_path` (str, optional): Path to MJCF XML file. If None, creates procedural model.

**Attributes:**
- `model` (MjModel): MuJoCo model
- `data` (MjData): MuJoCo data structure
- `nv` (int): Number of velocity DOFs
- `nq` (int): Number of position coordinates
- `nu` (int): Number of actuated joints

**Methods:**

#### `compute_nominal_dynamics(q, v)`

Extract analytical dynamics at given state.

```python
M, h, p_feet, J_feet = sim.compute_nominal_dynamics(q, v)
```

**Parameters:**
- `q` (ndarray): Generalized positions (nq,)
- `v` (ndarray): Generalized velocities (nv,)

**Returns:**
- `M` (ndarray): Mass matrix (nv, nv)
- `h` (ndarray): Nonlinear bias forces (nv,)
- `p_feet` (ndarray): Foot positions (4, 3)
- `J_feet` (ndarray): Foot Jacobians (4, 3, nv)

---

### Integrators

#### Function: `pind.utils.rk4_step`

4th-order Runge-Kutta integration step.

```python
from pind.utils import rk4_step

x_next = rk4_step(
    vector_field=dynamics_fn,
    x=x,
    u=u,
    dt=0.02
)
```

**Parameters:**
- `vector_field` (callable): Function f(x, u) returning ẋ
- `x` (Tensor): Current state (batch, state_dim)
- `u` (Tensor): Control input (batch, control_dim)
- `dt` (float): Time step
- `**kwargs`: Additional arguments for vector_field

**Returns:**
- `x_next` (Tensor): Next state (batch, state_dim)

---

## Data

### Data Generation

#### Function: `pind.data.collect_trajectories`

Generate training data from MuJoCo simulation.

```python
from pind.data import collect_trajectories

collect_trajectories(
    num_trajectories=20,
    trajectory_length=500,
    dt=0.02,
    save_path="data/dataset.pt",
    xml_path=None,     # Optional custom robot
    seed=42            # Random seed
)
```

**Parameters:**
- `num_trajectories` (int): Number of trajectories to collect
- `trajectory_length` (int): Steps per trajectory
- `dt` (float): Simulation timestep (seconds)
- `save_path` (str): Output path for dataset
- `xml_path` (str, optional): Custom robot MJCF
- `seed` (int, optional): Random seed for reproducibility

---

### Dataset Class

#### Class: `pind.data.QuadrupedDataset`

PyTorch Dataset for quadruped dynamics.

```python
from pind.data import QuadrupedDataset
from torch.utils.data import DataLoader

dataset = QuadrupedDataset("data/dataset.pt")
loader = DataLoader(dataset, batch_size=64, shuffle=True)

for batch in loader:
    q = batch['q']          # Positions
    v = batch['v']          # Velocities
    u = batch['u']          # Controls
    v_dot_gt = batch['v_dot_gt']  # Ground truth accelerations
    # ... use batch
```

**Sample Structure:**

Each sample contains:
```python
{
    'q': Tensor,           # Positions (19,)
    'v': Tensor,           # Velocities (18,)
    'u': Tensor,           # Controls (12,)
    'q_next': Tensor,      # Next positions (19,)
    'v_next': Tensor,      # Next velocities (18,)
    'v_dot_gt': Tensor,    # Ground truth acceleration (18,)
    'M_nom': Tensor,       # Mass matrix (18, 18)
    'h_nom': Tensor,       # Bias forces (18,)
    'p_feet': Tensor,      # Foot positions (4, 3)
    'J_feet': Tensor,      # Foot Jacobians (4, 3, 18)
}
```

---

## Training

### Training Script

```bash
python scripts/train.py \
    --config pind/configs/default.yaml \
    --data-path data/dataset.pt \
    --output-dir experiments/run1 \
    --model pind \
    --data-fraction 0.10 \
    --epochs 30 \
    --seed 42
```

**Arguments:**
- `--config`: Path to YAML config file
- `--data-path`: Path to dataset (.pt file)
- `--output-dir`: Directory for outputs
- `--model`: Model type (pind, mlp, node, residual)
- `--data-fraction`: Fraction of data to use (0.0-1.0)
- `--epochs`: Number of training epochs
- `--seed`: Random seed

---

## Evaluation

### Evaluation Script

```bash
python scripts/evaluate.py \
    --checkpoint experiments/run1/best_model.pt \
    --data-path data/dataset.pt \
    --horizon 50 \
    --num-eval 20 \
    --output results.csv
```

**Arguments:**
- `--checkpoint`: Path to trained model checkpoint
- `--data-path`: Path to evaluation dataset
- `--horizon`: Rollout horizon (timesteps)
- `--num-eval`: Number of evaluation rollouts
- `--output`: Path to save results CSV

**Output Metrics:**
- `position_rmse_mm`: Position prediction error (mm)
- `velocity_rmse`: Velocity prediction error
- `cone_violation_mean`: Friction cone violations (N)
- `penetration_mean_mm`: Ground penetration (mm)

---

## Configuration

### YAML Configuration File

```yaml
model:
  name: pind
  nv: 18
  nu: 12
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
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1

logging:
  log_interval: 10
  checkpoint_interval: 5
  use_tensorboard: true

system:
  device: cuda
  num_workers: 4
  seed: 42
```

---

## Examples

### Complete Training Pipeline

```python
import torch
from pind.models import PIND
from pind.data import QuadrupedDataset
from torch.utils.data import DataLoader

# Load data
dataset = QuadrupedDataset("data/dataset.pt")
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Create model
model = PIND(nv=18, nu=12, hidden_dim=128).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(30):
    for batch in loader:
        # Move to device
        q = batch['q'].cuda()
        v = batch['v'].cuda()
        u = batch['u'].cuda()
        v_dot_gt = batch['v_dot_gt'].cuda()
        M_nom = batch['M_nom'].cuda()
        h_nom = batch['h_nom'].cuda()
        p_feet = batch['p_feet'].cuda()
        J_feet = batch['J_feet'].cuda()
        
        # Forward pass
        v_dot, lambdas, delta_tau = model.forward_acceleration(
            q, v, u, M_nom, h_nom, p_feet, J_feet
        )
        
        # Compute loss
        loss, metrics = model.compute_composite_loss(
            v_dot, v_dot_gt, lambdas, p_feet,
            M_nom, h_nom, u, delta_tau, J_feet
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    
    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

---

## Version Information

- **Package Version**: 0.1.0
- **PyTorch Version**: >= 2.0.0
- **Python Version**: >= 3.8
- **MuJoCo Version**: >= 3.0.0

---

## Troubleshooting

### Common Issues

**Q: CUDA out of memory**
```python
# Reduce batch size in config
training:
  batch_size: 32  # Default is 64
```

**Q: MuJoCo import error**
```bash
pip install --upgrade mujoco
python -c "import mujoco; print(mujoco.__version__)"
```

**Q: Slow training**
- Use GPU if available
- Reduce `num_workers` in DataLoader
- Use smaller `data_fraction` for testing
