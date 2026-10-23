# Experiments Directory

This directory stores all experiment outputs including model checkpoints, training logs, and evaluation results.

## 📁 Directory Structure

```
experiments/
├── README.md                          # This file
├── logs/                              # TensorBoard training logs
├── checkpoints/                       # Model checkpoints during training
├── results/                           # Final evaluation results (CSV, plots)
└── <run_timestamp>/                   # Individual experiment runs
    ├── best_model.pt                  # Best model checkpoint
    ├── config.yaml                    # Experiment configuration
    ├── logs/                          # TensorBoard logs for this run
    └── checkpoint_epoch_*.pt          # Periodic checkpoints
```

## 🎯 What Gets Stored Here

### 1. Training Runs
When you run training, a timestamped directory is created:
```bash
python scripts/train.py --config pind/configs/default.yaml
```

Output:
```
experiments/pind_20241023_162830/
├── best_model.pt
├── config.yaml
├── logs/
└── checkpoint_epoch_5.pt
```

### 2. Benchmark Results
When you run the full benchmark:
```bash
python run_experiments.py
```

Output:
```
experiments/results/benchmark_20241023_163000/
├── benchmark_results.csv
├── experiment_config.yaml
├── data_5pct/
│   ├── seed_1/model_seed1.pt
│   ├── seed_2/model_seed2.pt
│   └── ...
├── data_10pct/
└── data_100pct/
```

## 📊 Accessing Results

### View Training Logs
```bash
# Launch TensorBoard
tensorboard --logdir experiments/

# Or for specific run
tensorboard --logdir experiments/pind_20241023_162830/logs
```

### Load Trained Model
```python
import torch

# Load checkpoint
checkpoint = torch.load('experiments/pind_20241023_162830/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Read Benchmark Results
```bash
# View CSV results
cat experiments/results/benchmark_*/benchmark_results.csv

# Or with pandas
python -c "import pandas as pd; print(pd.read_csv('experiments/results/benchmark_*/benchmark_results.csv'))"
```

## 🧹 Cleaning Up

### Remove Old Experiments
```bash
# Remove all experiments (be careful!)
rm -rf experiments/*/

# Remove specific run
rm -rf experiments/pind_20241023_162830/
```

### Keep Results, Remove Checkpoints
```bash
# Large checkpoint files can be removed while keeping logs
find experiments/ -name "*.pt" -type f -delete
```

## 📏 Expected Sizes

- **Single training run**: 50-200 MB (depends on model size)
- **Full benchmark (5 seeds, 4 regimes)**: 1-4 GB
- **TensorBoard logs**: 10-50 MB per run

## ⚠️ Important Notes

1. **This directory is in .gitignore** - outputs won't be committed to git
2. **Experiments are timestamped** - each run creates a unique directory
3. **Checkpoints can be large** - clean periodically
4. **Results are portable** - can be copied/shared independently

## 🔍 Example Workflow

```bash
# 1. Train model
python scripts/train.py --output-dir experiments/my_experiment

# 2. Check results
ls experiments/my_experiment/

# 3. Evaluate
python scripts/evaluate.py \
    --checkpoint experiments/my_experiment/best_model.pt \
    --output experiments/my_experiment/eval_results.csv

# 4. View logs
tensorboard --logdir experiments/my_experiment/logs

# 5. Share results
cp experiments/my_experiment/best_model.pt ./final_model.pt
cp experiments/results/benchmark_results.csv ./paper_results.csv
```

## 📦 Backup Recommendations

For important experiments, backup:
- ✅ `best_model.pt` - Best trained model
- ✅ `config.yaml` - Exact configuration used
- ✅ `*.csv` - Evaluation metrics
- ⚠️ Checkpoints - Only if needed for resuming
- ⚠️ Logs - Only if analyzing training curves

---

**This directory will be automatically populated when you run experiments!**

No manual setup needed - just start training! 🚀
