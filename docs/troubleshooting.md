# Troubleshooting Guide

Common issues and solutions when using PIND.

## 📋 Table of Contents

- [Installation Issues](#installation-issues)
- [Training Problems](#training-problems)
- [Evaluation Issues](#evaluation-issues)
- [Performance Issues](#performance-issues)
- [Data Generation Problems](#data-generation-problems)
- [Hardware/Environment Issues](#hardwareenvironment-issues)

---

## Installation Issues

### Problem: `pip install -e .` fails

**Solution 1: Update pip**
```bash
pip install --upgrade pip setuptools wheel
pip install -e .
```

**Solution 2: Install dependencies first**
```bash
pip install -r requirements.txt
pip install -e .
```

**Solution 3: Use conda environment**
```bash
conda create -n pind python=3.10
conda activate pind
pip install -e .
```

---

### Problem: MuJoCo not found

**Error:**
```
ModuleNotFoundError: No module named 'mujoco'
```

**Solution:**
```bash
pip install mujoco>=3.0.0

# Verify installation
python -c "import mujoco; print(f'MuJoCo {mujoco.__version__} installed')"
```

---

### Problem: PyTorch CUDA mismatch

**Error:**
```
RuntimeError: CUDA error: invalid device ordinal
```

**Solution:**
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch
# For CUDA 11.8:
pip install torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# For CUDA 12.1:
pip install torch==2.0.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html

# For CPU only:
pip install torch==2.0.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

---

## Training Problems

### Problem: CUDA out of memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution 1: Reduce batch size**
```yaml
# Edit pind/configs/default.yaml
training:
  batch_size: 32  # Try 32, 16, or even 8
```

**Solution 2: Use gradient accumulation**
```python
# In training script, accumulate gradients over multiple batches
accumulation_steps = 4
for i, batch in enumerate(loader):
    loss = compute_loss(...)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Solution 3: Use CPU**
```bash
python scripts/train.py --device cpu
```

**Solution 4: Clear CUDA cache**
```python
import torch
torch.cuda.empty_cache()
```

---

### Problem: Training loss is NaN

**Possible Causes:**
1. Learning rate too high
2. Numerical instability
3. Gradient explosion

**Solution 1: Lower learning rate**
```yaml
training:
  learning_rate: 0.0001  # Try 10× smaller
```

**Solution 2: Enable gradient clipping**
```yaml
training:
  grad_clip: 1.0  # Try smaller values
```

**Solution 3: Check for NaN inputs**
```python
# Add checks in training loop
assert not torch.isnan(M_nom).any(), "NaN in M_nom"
assert not torch.isnan(h_nom).any(), "NaN in h_nom"
```

**Solution 4: Use mixed precision (carefully)**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for batch in loader:
    with autocast():
        loss = compute_loss(...)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

### Problem: Loss not decreasing

**Solution 1: Check data**
```python
# Verify dataset loads correctly
dataset = QuadrupedDataset("data/dataset.pt")
sample = dataset[0]
print({k: v.shape for k, v in sample.items()})
```

**Solution 2: Start with supervised loss only**
```yaml
loss_weights:
  beta_dyn: 0.0    # Disable physics losses temporarily
  beta_cone: 0.0
  beta_pen: 0.0
```

**Solution 3: Try different optimizer**
```python
# Instead of AdamW, try SGD or Adam
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

**Solution 4: Check learning rate schedule**
```python
# Add learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

---

### Problem: Training crashes randomly

**Solution 1: Pin memory off**
```yaml
system:
  pin_memory: false
```

**Solution 2: Reduce num_workers**
```yaml
system:
  num_workers: 0  # Single-process data loading
```

**Solution 3: Check disk space**
```bash
df -h  # Ensure enough space for checkpoints
```

---

## Evaluation Issues

### Problem: Checkpoint not found

**Error:**
```
FileNotFoundError: experiments/*/best_model.pt not found
```

**Solution:**
```bash
# Find actual checkpoint path
find experiments/ -name "best_model.pt"

# Use explicit path
python scripts/evaluate.py --checkpoint experiments/pind_20241023_*/best_model.pt
```

---

### Problem: Evaluation gives poor results

**Possible Causes:**
1. Model not fully trained
2. Train/eval distribution mismatch
3. Wrong model loaded

**Solution 1: Check training completed**
```bash
# Verify checkpoint contains expected epochs
python -c "import torch; ckpt = torch.load('path/to/checkpoint.pt'); print(f\"Epoch: {ckpt.get('epoch', 'unknown')}\")"
```

**Solution 2: Evaluate on training data first**
```bash
# Should give good results if model trained properly
python scripts/evaluate.py --checkpoint ... --data-path data/train_dataset.pt
```

**Solution 3: Check model architecture matches**
```python
# Ensure config matches training config
checkpoint = torch.load('checkpoint.pt')
print(checkpoint['config'])
```

---

## Performance Issues

### Problem: Training too slow

**Solution 1: Use GPU**
```bash
# Verify GPU is being used
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Solution 2: Increase batch size (if memory allows)**
```yaml
training:
  batch_size: 128  # Larger batches = better GPU utilization
```

**Solution 3: Increase num_workers**
```yaml
system:
  num_workers: 8  # Match number of CPU cores
```

**Solution 4: Use smaller dataset for testing**
```bash
python scripts/train.py --data-fraction 0.01 --epochs 5
```

---

### Problem: Data generation slow

**Solution 1: Reduce trajectory length**
```bash
python -m pind.data.generator --trajectory-length 250  # Instead of 500
```

**Solution 2: Use fewer trajectories**
```bash
python -m pind.data.generator --num-trajectories 10  # Instead of 20
```

**Solution 3: Increase timestep (less accurate but faster)**
```bash
python -m pind.data.generator --dt 0.05  # Instead of 0.02
```

---

## Data Generation Problems

### Problem: MuJoCo simulation crashes

**Error:**
```
Segmentation fault (core dumped)
```

**Solution 1: Update MuJoCo**
```bash
pip install --upgrade mujoco
```

**Solution 2: Check XML path**
```bash
# Verify custom XML is valid
python -c "import mujoco; model = mujoco.MjModel.from_xml_path('your_robot.xml')"
```

**Solution 3: Use procedural model**
```bash
# Don't specify --xml-path to use built-in model
python -m pind.data.generator
```

---

### Problem: Dataset file corrupted

**Error:**
```
RuntimeError: PytorchStreamReader failed reading zip archive
```

**Solution:**
```bash
# Delete and regenerate
rm data/go1_dataset.pt
make data
```

---

## Hardware/Environment Issues

### Problem: GPU not detected

**Solution 1: Check NVIDIA driver**
```bash
nvidia-smi
```

**Solution 2: Verify CUDA installation**
```bash
nvcc --version
```

**Solution 3: Reinstall PyTorch with CUDA**
```bash
pip uninstall torch
pip install torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

---

### Problem: Running out of disk space

**Solution 1: Clean old experiments**
```bash
# Remove old experiment directories
rm -rf experiments/pind_2024*/

# Keep only best models
find experiments/ -name "checkpoint_epoch_*.pt" -delete
```

**Solution 2: Use smaller checkpoints**
```yaml
logging:
  checkpoint_interval: 10  # Save less frequently
```

**Solution 3: Disable checkpoint saving**
```yaml
logging:
  checkpoint_interval: 0  # Only save best model
```

---

### Problem: Insufficient RAM

**Solution 1: Reduce data workers**
```yaml
system:
  num_workers: 0
```

**Solution 2: Load data in smaller chunks**
```python
# Use dataset subset
from torch.utils.data import Subset
subset = Subset(dataset, range(0, len(dataset), 2))  # Use every 2nd sample
```

**Solution 3: Use swap space**
```bash
# Add swap (Linux)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## Docker Issues

### Problem: Docker container won't start

**Solution 1: Check Docker is running**
```bash
docker ps
```

**Solution 2: Rebuild image**
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up
```

**Solution 3: Check permissions**
```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
# Log out and back in
```

---

### Problem: GPU not available in Docker

**Solution 1: Use nvidia-docker**
```bash
# Install nvidia-docker2
sudo apt-get install nvidia-docker2
sudo systemctl restart docker
```

**Solution 2: Add runtime to docker-compose**
```yaml
services:
  pind:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

---

## Still Having Issues?

### Getting Help

1. **Check existing issues**: [GitHub Issues](https://github.com/imsanjoykb/PIND-Quadruped/issues)

2. **Open a new issue**: Include:
   - Operating system and version
   - Python version (`python --version`)
   - PyTorch version (`python -c "import torch; print(torch.__version__)"`)
   - CUDA version (if applicable, `nvidia-smi`)
   - MuJoCo version (`python -c "import mujoco; print(mujoco.__version__)"`)
   - Full error message and stack trace
   - Minimal code to reproduce the issue

3. **Start a discussion**: [GitHub Discussions](https://github.com/imsanjoykb/PIND-Quadruped/discussions)

4. **Community support**:
   - Check [README.md](../README.md) for common issues
   - Review [API documentation](api.md)
   - See [Quick Start Guide](quickstart.md)

---

## Debugging Tips

### Enable verbose logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check intermediate values

```python
# Add print statements in training loop
print(f"q shape: {q.shape}, v shape: {v.shape}")
print(f"M_nom min: {M_nom.min()}, max: {M_nom.max()}")
print(f"Loss: {loss.item()}")
```

### Use debugger

```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or use iPython debugger
import ipdb; ipdb.set_trace()
```

### Monitor GPU usage

```bash
# In separate terminal
watch -n 1 nvidia-smi
```

### Profile code

```python
import cProfile
import pstats

with cProfile.Profile() as pr:
    # Your training code here
    pass

stats = pstats.Stats(pr)
stats.sort_stats('cumtime')
stats.print_stats(20)
```

---

## Preventing Common Issues

### Best Practices

1. **Always use virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Pin dependency versions**
   ```bash
   pip freeze > requirements.txt
   ```

3. **Use version control**
   ```bash
   git add . && git commit -m "Checkpoint before experiment"
   ```

4. **Save configurations**
   ```bash
   # Configs are automatically saved in experiment directories
   ```

5. **Monitor training**
   ```bash
   tensorboard --logdir experiments/
   ```

6. **Start small, scale up**
   ```bash
   # Test with tiny dataset first
   python scripts/train.py --data-fraction 0.01 --epochs 3
   ```

---

**Last Updated**: October 2024

For additional help, please open an issue on GitHub or consult the documentation.
