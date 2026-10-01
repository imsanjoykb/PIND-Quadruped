import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse

sys.path.insert(0, str(Path(__file__).parent))

from scripts.train import create_model, train_epoch, validate, collate_fn
from scripts.evaluate import evaluate_multi_step_rollout
from pind.data.generator import collect_trajectories, QuadrupedDataset
from torch.utils.data import DataLoader, Subset


def run_single_experiment(config, data_path, data_fraction, seed, output_dir, device):

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Update config
    config['data']['data_fraction'] = data_fraction
    config['system']['seed'] = seed
    
    # Load dataset
    dataset = QuadrupedDataset(data_path)
    n_total = len(dataset)
    n_train = int(n_total * data_fraction * config['data']['train_split'])
    
    indices = np.random.permutation(int(n_total * data_fraction))[:n_train]
    train_dataset = Subset(dataset, indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Create model
    model = create_model(config, device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Train
    print(f"  Training with {len(train_dataset)} samples (seed {seed})...")
    for epoch in range(1, config['training']['epochs'] + 1):
        train_loss, _ = train_epoch(
            model, train_loader, optimizer, config, device, epoch
        )
        if epoch % 5 == 0:
            print(f"    Epoch {epoch}/{config['training']['epochs']}: Loss {train_loss:.6f}")
    
    # Evaluate
    print(f"  Evaluating...")
    metrics = evaluate_multi_step_rollout(
        model, dataset, config, horizon=50, num_eval=20, device=device
    )
    
    # Save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'metrics': metrics
    }, output_dir / f"model_seed{seed}.pt")
    
    return metrics


def run_reproducible_benchmark(config_path='pind/configs/data_efficiency.yaml'):
    """Run full reproducible benchmark across seeds and data regimes."""
    # Load experiment config
    with open(config_path, 'r') as f:
        exp_config = yaml.safe_load(f)
    
    # Load base model config
    base_config_path = 'pind/configs/default.yaml'
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    data_path = "data/go1_dataset.pt"
    
    # Generate data if needed
    if not os.path.exists(data_path):
        print("Generating training data...")
        collect_trajectories(
            num_trajectories=20,
            trajectory_length=500,
            save_path=data_path
        )
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(exp_config['output']['results_dir']) / f"benchmark_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*80}")
    print(f"STARTING BENCHMARK: {exp_config['experiment']['name']}")
    print(f"{'='*80}\n")
    
    # Run experiments
    all_results = []
    
    for data_fraction in exp_config['data_regimes']:
        print(f"\n{'='*80}")
        print(f"Data Regime: {int(data_fraction * 100)}%")
        print(f"{'='*80}\n")
        
        regime_results = []
        
        for seed in exp_config['seeds']:
            # Update config
            config = base_config.copy()
            config['training']['epochs'] = exp_config['training']['epochs']
            config['training']['batch_size'] = exp_config['training']['batch_size']
            config['training']['learning_rate'] = exp_config['training']['learning_rate']
            
            # Run experiment
            regime_dir = output_dir / f"data_{int(data_fraction*100)}pct" / f"seed_{seed}"
            regime_dir.mkdir(parents=True, exist_ok=True)
            
            metrics = run_single_experiment(
                config, data_path, data_fraction, seed, regime_dir, device
            )
            
            regime_results.append(metrics)
            print(f"  Seed {seed} - Pos RMSE: {metrics['position_rmse_mm']:.2f} mm, " 
                  f"Cone Viol: {metrics['cone_violation_mean']:.4f} N\n")
        
        # Aggregate results
        pos_rmse = [r['position_rmse_mm'] for r in regime_results]
        cone_viol = [r['cone_violation_mean'] for r in regime_results]
        
        all_results.append({
            'Data Fraction': f"{int(data_fraction * 100)}%",
            'Position RMSE (mm) Mean': np.mean(pos_rmse),
            'Position RMSE (mm) Std': np.std(pos_rmse),
            'Cone Violation (N) Mean': np.mean(cone_viol),
            'Cone Violation (N) Std': np.std(cone_viol)
        })
    
    # Create results table
    df = pd.DataFrame(all_results)
    
    print(f"\n{'='*80}")
    print("FINAL BENCHMARK RESULTS")
    print(f"{'='*80}\n")
    print(df.to_string(index=False))
    print(f"\n{'='*80}\n")
    
    # Save results
    results_path = output_dir / 'benchmark_results.csv'
    df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")
    
    # Save config
    with open(output_dir / 'experiment_config.yaml', 'w') as f:
        yaml.dump(exp_config, f)
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Run reproducible benchmark experiments")
    parser.add_argument('--config', type=str, default='pind/configs/data_efficiency.yaml')
    args = parser.parse_args()
    
    run_reproducible_benchmark(args.config)


if __name__ == "__main__":
    main()
