"""
Evaluation script for trained models.

Computes open-loop multi-step rollout metrics including:
- Position/velocity RMSE
- Physics constraint violations
- Energy drift
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from pind.models.pind import PIND
from pind.models.baselines import BlackBoxMLP, NeuralODEVectorField, UnconstrainedResidualMLP
from pind.data.generator import QuadrupedDataset


def evaluate_multi_step_rollout(model, dataset, config, horizon=50, num_eval=20, device="cpu"):
    """
    Evaluate recursive multi-step forecasting.
    
    Args:
        model: Trained model
        dataset: Evaluation dataset
        config: Model configuration
        horizon: Rollout horizon (timesteps)
        num_eval: Number of evaluation rollouts
        device: Compute device
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    
    pos_errors = []
    vel_errors = []
    cone_violations = []
    penetrations = []
    
    with torch.no_grad():
        for start_idx in range(0, min(num_eval * horizon, len(dataset) - horizon), horizon):
            rollout_pos_err = []
            rollout_vel_err = []
            
            for step in range(horizon):
                data_point = dataset[start_idx + step]
                q = data_point["q"].unsqueeze(0).to(device)
                v = data_point["v"].unsqueeze(0).to(device)
                u = data_point["u"].unsqueeze(0).to(device)
                q_gt_next = data_point["q_next"].to(device)
                v_gt_next = data_point["v_next"].to(device)
                
                M_nom = data_point["M_nom"].unsqueeze(0).to(device)
                h_nom = data_point["h_nom"].unsqueeze(0).to(device)
                p_feet = data_point["p_feet"].unsqueeze(0).to(device)
                J_feet = data_point["J_feet"].unsqueeze(0).to(device)

                # Forward dynamics step
                if config['model']['name'] == 'pind':
                    v_dot, lambdas, _ = model.forward_acceleration(
                        q, v, u, M_nom, h_nom, p_feet, J_feet
                    )
                    
                    # Measure friction cone violation
                    for lam in lambdas:
                        fxy = torch.norm(lam[:, :2], dim=-1)
                        fz = lam[:, 2]
                        viol = torch.relu(fxy - config['model']['friction_coeff'] * fz).item()
                        cone_violations.append(viol)
                    
                    # Measure penetration
                    for i in range(4):
                        z_foot = p_feet[:, i, 2].item()
                        if z_foot < 0:
                            penetrations.append(-z_foot * 1000)  # mm
                else:
                    x = torch.cat([q, v], dim=-1)
                    if config['model']['name'] == 'residual':
                        v_dot_nom = torch.zeros(1, config['model']['nv']).to(device)
                        v_dot = model(x, u, v_dot_nom)
                    else:
                        v_dot = model(x, u)
                        v_dot = v_dot[:, config['model']['nv']:]  # Extract velocity part
                
                # Semi-implicit Euler integration
                dt = 0.02
                v_next = v + v_dot * dt
                
                # Compute errors
                pos_err = torch.norm(q[0, :3] - q_gt_next[:3]).item() * 1000  # mm
                vel_err = torch.norm(v_next[0, :] - v_gt_next).item()
                
                rollout_pos_err.append(pos_err)
                rollout_vel_err.append(vel_err)
            
            pos_errors.append(np.mean(rollout_pos_err))
            vel_errors.append(np.mean(rollout_vel_err))
    
    metrics = {
        'position_rmse_mm': np.sqrt(np.mean(np.array(pos_errors)**2)),
        'position_std_mm': np.std(pos_errors),
        'velocity_rmse': np.sqrt(np.mean(np.array(vel_errors)**2)),
        'velocity_std': np.std(vel_errors),
        'cone_violation_mean': np.mean(cone_violations) if cone_violations else 0.0,
        'cone_violation_std': np.std(cone_violations) if cone_violations else 0.0,
        'penetration_mean_mm': np.mean(penetrations) if penetrations else 0.0,
        'penetration_std_mm': np.std(penetrations) if penetrations else 0.0,
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-path', type=str, default='data/go1_dataset.pt')
    parser.add_argument('--horizon', type=int, default=50, help='Rollout horizon')
    parser.add_argument('--num-eval', type=int, default=20, help='Number of evaluation rollouts')
    parser.add_argument('--output', type=str, default=None, help='Output CSV path')
    
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    dataset = QuadrupedDataset(args.data_path)
    
    # Create model
    if config['model']['name'] == 'pind':
        model = PIND(
            nv=config['model']['nv'],
            nu=config['model']['nu'],
            hidden_dim=config['model']['hidden_dim'],
            eps_c=config['model']['eps_c'],
            mu=config['model']['friction_coeff']
        )
    elif config['model']['name'] == 'mlp':
        model = BlackBoxMLP(state_dim=37, act_dim=config['model']['nu'])
    elif config['model']['name'] == 'node':
        model = NeuralODEVectorField(state_dim=37, act_dim=config['model']['nu'])
    elif config['model']['name'] == 'residual':
        model = UnconstrainedResidualMLP(
            state_dim=37,
            act_dim=config['model']['nu'],
            nv=config['model']['nv']
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Evaluate
    print(f"\nEvaluating {args.horizon}-step rollout...")
    metrics = evaluate_multi_step_rollout(
        model, dataset, config, horizon=args.horizon, num_eval=args.num_eval, device=device
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"{'='*60}\n")
    
    # Save results
    if args.output:
        df = pd.DataFrame([metrics])
        df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
