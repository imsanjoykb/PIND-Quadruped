# 🚀 Get Started with PIND-Quadruped

Welcome! This guide will get you up and running in 5 minutes.

## ⚡ Quick Setup (3 steps)

### 1️⃣ Install

```bash
# Clone and enter directory
cd /Users/imsanjoykb/ResearchWork/PIND-Quadruped

# Install package
pip install -e .
```

### 2️⃣ Generate Data

```bash
# Generate 20 trajectories (~2-3 minutes)
make data

# Or manually:
python -m pind.data.generator --num-trajectories 20 --output data/go1_dataset.pt
```

### 3️⃣ Train Model

```bash
# Quick test (10% data, 10 epochs, ~5 minutes)
make train

# Or full training:
make train-full
```

## 📊 Verify Installation

```bash
./verify_setup.sh
```

## 🎯 What's Next?

### Option A: Quick Experiment
```bash
# Run a 5-minute test
python scripts/train.py --data-fraction 0.01 --epochs 3
```

### Option B: Full Benchmark
```bash
# Run complete experiments (4-6 hours on GPU)
make benchmark
```

### Option C: Explore the Code
```bash
# Read the comprehensive README
cat README.md

# Check quick start guide
cat docs/quickstart.md

# View available commands
make help
```

## 📁 Repository Structure

```
PIND-Quadruped/
├── pind/               # Main package
│   ├── models/         # PIND + baselines
│   ├── utils/          # MuJoCo interface
│   ├── data/           # Data generation
│   └── configs/        # YAML configs
├── scripts/            # train.py, evaluate.py
├── tests/              # Unit tests
├── docs/               # Documentation
└── experiments/        # Output directory
```

## 🛠️ Useful Commands

```bash
make help              # Show all commands
make test              # Run tests
make format            # Format code
make clean             # Clean temp files
make tensorboard       # Launch TensorBoard
make docker-build      # Build Docker image
```

## 🔍 Check Results

After training, check:
```bash
# View tensorboard logs
tensorboard --logdir experiments/

# Check saved models
ls experiments/*/best_model.pt

# View benchmark results
cat experiments/results/benchmark_results.csv
```

## 📚 Documentation

- **README.md** - Full project documentation
- **docs/quickstart.md** - Detailed quick start
- **CONTRIBUTING.md** - How to contribute
- **PROJECT_SUMMARY.md** - Complete feature list

## 🐛 Troubleshooting

### Import Error
```bash
pip install -e .
```

### CUDA Out of Memory
Reduce batch size in `pind/configs/default.yaml`:
```yaml
training:
  batch_size: 32  # Default is 64
```

### Slow Training
- Use GPU if available
- Use `--data-fraction 0.1` for quick tests
- Reduce `--epochs` for testing

## 💡 Tips

1. **Start small**: Use `--data-fraction 0.01` for quick tests
2. **Use Makefile**: All common commands are in `make help`
3. **Check configs**: Customize `pind/configs/default.yaml`
4. **Read docs**: `docs/quickstart.md` has detailed examples

## 🎓 Research Use

For paper experiments:
```bash
# Run data efficiency benchmark (Table II)
python run_experiments.py --config pind/configs/data_efficiency.yaml

# Results will be in:
# experiments/results/benchmark_results.csv
```

## 📧 Help

- GitHub Issues: Report bugs
- README.md: Full documentation
- docs/: Detailed guides

---

**Ready to start?** Run: `make install && make data && make train`

🎉 Happy researching!
