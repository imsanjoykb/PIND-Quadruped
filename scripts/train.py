import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pind.models.pind import PIND
from pind.models.baselines import BlackBoxMLP, NeuralODEVectorField, UnconstrainedResidualMLP
from pind.data.generator import QuadrupedDataset


def collate_fn(batch):
    """Custom collate function for dataloader."""
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0].keys()}


def create_model(config, device):
    """Create model based on configuration."""
    model_name = config['model']['name']
    
    if model_name == 'pind':
        model = PIND(
            nv=config['model']['nv'],
            nu=config['model']['nu'],
            hidden_dim=config['model']['hidden_dim'],
            eps_c=config['model']['eps_c'],
            mu=config['model']['friction_coeff']
        )
    elif model_name == 'mlp':
        model = BlackBoxMLP(
            state_dim=37,
            act_dim=config['model']['nu'],
            hidden_dim=config['model'].get('hidden_dim', 256)
        )
    elif model_name == 'node':
        model = NeuralODEVectorField(
            state_dim=37,
            act_dim=config['model']['nu'],
            hidden_dim=config['model'].get('hidden_dim', 256)
        )
    elif model_name == 'residual':
        model = UnconstrainedResidualMLP(
            state_dim=37,
            act_dim=config['model']['nu'],
            nv=config['model']['nv'],
            hidden_dim=config['model']['hidden_dim']
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model.to(device)


def train_epoch(model, loader, optimizer, config, device, epoch, writer=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    metrics_sum = {
        'l_data': 0.0,
        'l_dyn': 0.0,
        'l_cone': 0.0,
        'l_pen': 0.0
    }
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        q = batch["q"].to(device)
        v = batch["v"].to(device)
        u = batch["u"].to(device)
        v_dot_gt = batch["v_dot_gt"].to(device)
        M_nom = batch["M_nom"].to(device)
        h_nom = batch["h_nom"].to(device)
        p_feet = batch["p_feet"].to(device)
        J_feet = batch["J_feet"].to(device)

        optimizer.zero_grad()
        
        # Forward pass
        if config['model']['name'] == 'pind':
            v_dot_pred, lambdas, delta_tau = model.forward_acceleration(
                q, v, u, M_nom, h_nom, p_feet, J_feet
            )
            
            loss, metrics = model.compute_composite_loss(
                v_dot_pred, v_dot_gt, lambdas, p_feet, M_nom, h_nom, u, delta_tau, J_feet,
                beta_dyn=config['loss_weights']['beta_dyn'],
                beta_cone=config['loss_weights']['beta_cone'],
                beta_pen=config['loss_weights']['beta_pen'],
                alpha_comp=config['loss_weights']['alpha_comp']
            )
            
            for k in metrics_sum:
                metrics_sum[k] += metrics[k]
        else:
            # Baseline models - simple MSE loss
            if config['model']['name'] == 'residual':
                v_dot_nom = torch.zeros_like(v_dot_gt)  # Simplified
                x = torch.cat([q, v], dim=-1)
                v_dot_pred = model(x, u, v_dot_nom)
            else:
                x = torch.cat([q, v], dim=-1)
                v_dot_pred = model(x, u)
            
            loss = nn.functional.mse_loss(v_dot_pred, v_dot_gt)
            metrics = {'l_data': loss.item()}
        
        # Backward pass
        loss.backward()
        if config['training'].get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['training']['grad_clip']
            )
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
        
        # Log to tensorboard
        if writer and batch_idx % config['logging']['log_interval'] == 0:
            global_step = epoch * len(loader) + batch_idx
            writer.add_scalar('train/loss', loss.item(), global_step)
            if config['model']['name'] == 'pind':
                for k, v in metrics.items():
                    writer.add_scalar(f'train/{k}', v, global_step)
    
    avg_loss = total_loss / len(loader)
    avg_metrics = {k: v / len(loader) for k, v in metrics_sum.items()}
    
    return avg_loss, avg_metrics


def validate(model, loader, config, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            q = batch["q"].to(device)
            v = batch["v"].to(device)
            u = batch["u"].to(device)
            v_dot_gt = batch["v_dot_gt"].to(device)
            M_nom = batch["M_nom"].to(device)
            h_nom = batch["h_nom"].to(device)
            p_feet = batch["p_feet"].to(device)
            J_feet = batch["J_feet"].to(device)
            
            if config['model']['name'] == 'pind':
                v_dot_pred, lambdas, delta_tau = model.forward_acceleration(
                    q, v, u, M_nom, h_nom, p_feet, J_feet
                )
                loss, _ = model.compute_composite_loss(
                    v_dot_pred, v_dot_gt, lambdas, p_feet, M_nom, h_nom, u, delta_tau, J_feet,
                    beta_dyn=config['loss_weights']['beta_dyn'],
                    beta_cone=config['loss_weights']['beta_cone'],
                    beta_pen=config['loss_weights']['beta_pen']
                )
            else:
                x = torch.cat([q, v], dim=-1)
                if config['model']['name'] == 'residual':
                    v_dot_nom = torch.zeros_like(v_dot_gt)
                    v_dot_pred = model(x, u, v_dot_nom)
                else:
                    v_dot_pred = model(x, u)
                loss = nn.functional.mse_loss(v_dot_pred, v_dot_gt)
            
            total_loss += loss.item()
    
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Train PIND or baseline models")
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=str, default='data/go1_dataset.pt')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--model', type=str, default=None, help='Model name (overrides config)')
    parser.add_argument('--data-fraction', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config_path = Path(__file__).parent.parent / 'pind' / 'configs' / 'default.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Override config with command line args
    if args.model:
        config['model']['name'] = args.model
    if args.data_fraction:
        config['data']['data_fraction'] = args.data_fraction
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.seed:
        config['system']['seed'] = args.seed
    
    # Set random seeds
    torch.manual_seed(config['system']['seed'])
    np.random.seed(config['system']['seed'])
    
    # Setup device
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('experiments') / f"{config['model']['name']}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Setup logging
    if config['logging']['use_tensorboard']:
        writer = SummaryWriter(output_dir / 'logs')
    else:
        writer = None
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    dataset = QuadrupedDataset(args.data_path)
    
    # Split dataset
    n_total = len(dataset)
    n_train = int(n_total * config['data']['data_fraction'] * config['data']['train_split'])
    n_val = int(n_total * config['data']['data_fraction'] * config['data']['val_split'])
    
    indices = np.random.permutation(int(n_total * config['data']['data_fraction']))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['system']['num_workers'],
        pin_memory=config['system']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['system']['num_workers'],
        pin_memory=config['system']['pin_memory']
    )
    
    # Create model
    print(f"Creating {config['model']['name']} model...")
    model = create_model(config, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    print(f"\n{'='*60}")
    print(f"Starting training for {config['training']['epochs']} epochs")
    print(f"{'='*60}\n")
    
    for epoch in range(1, config['training']['epochs'] + 1):
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, config, device, epoch, writer
        )
        
        # Validate
        if epoch % config['evaluation']['eval_interval'] == 0:
            val_loss = validate(model, val_loader, config, device)
            
            print(f"\nEpoch {epoch}/{config['training']['epochs']}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            if config['model']['name'] == 'pind':
                print(f"  Data Loss: {train_metrics['l_data']:.6f}")
                print(f"  Dyn Residual: {train_metrics['l_dyn']:.6f}")
            
            if writer:
                writer.add_scalar('val/loss', val_loss, epoch)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': config
                }, output_dir / 'best_model.pt')
                print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")
        
        # Save checkpoint
        if epoch % config['logging']['checkpoint_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")
    print(f"{'='*60}\n")
    
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
