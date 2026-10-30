import torch
import numpy as np
from typing import Dict, List, Tuple


def compute_position_rmse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    timesteps: int = None
) -> float:
    """
    Compute position RMSE.
    
    Args:
        predicted: Predicted positions (batch, timesteps, 3) or (batch, 3)
        target: Target positions (batch, timesteps, 3) or (batch, 3)
        timesteps: Number of timesteps (if None, computed from shape)
    
    Returns:
        RMSE in millimeters
    """
    error = torch.norm(predicted - target, dim=-1)  # Euclidean distance
    rmse = torch.sqrt(torch.mean(error ** 2))
    return rmse.item() * 1000  # Convert to mm


def compute_velocity_rmse(
    predicted: torch.Tensor,
    target: torch.Tensor
) -> float:
    """
    Compute velocity RMSE.
    
    Args:
        predicted: Predicted velocities (batch, dim)
        target: Target velocities (batch, dim)
    
    Returns:
        RMSE value
    """
    error = torch.norm(predicted - target, dim=-1)
    rmse = torch.sqrt(torch.mean(error ** 2))
    return rmse.item()


def compute_friction_cone_violation(
    contact_forces: List[torch.Tensor],
    mu: float = 0.8
) -> Dict[str, float]:
    """
    Compute friction cone violations.
    
    Args:
        contact_forces: List of contact forces (batch, 3) for each foot
        mu: Friction coefficient
    
    Returns:
        Dictionary with mean and max violations
    """
    violations = []
    
    for forces in contact_forces:
        fx, fy, fz = forces[:, 0], forces[:, 1], forces[:, 2]
        f_tangent = torch.sqrt(fx**2 + fy**2)
        
        # Violation = max(0, ||f_tangent|| - mu * f_z)
        violation = torch.relu(f_tangent - mu * fz)
        violations.append(violation)
    
    all_violations = torch.cat(violations)
    
    return {
        'mean': all_violations.mean().item(),
        'std': all_violations.std().item(),
        'max': all_violations.max().item()
    }


def compute_penetration_depth(
    foot_positions: torch.Tensor,
    ground_height: float = 0.0
) -> Dict[str, float]:
    """
    Compute ground penetration depth.
    
    Args:
        foot_positions: Foot positions (batch, 4, 3)
        ground_height: Ground plane height
    
    Returns:
        Dictionary with penetration statistics
    """
    z_positions = foot_positions[:, :, 2]  # (batch, 4)
    
    # Penetration = max(0, ground_height - z)
    penetration = torch.relu(ground_height - z_positions)
    
    # Convert to mm
    penetration_mm = penetration * 1000
    
    return {
        'mean': penetration_mm.mean().item(),
        'std': penetration_mm.std().item(),
        'max': penetration_mm.max().item()
    }


def compute_energy_drift(
    initial_energy: torch.Tensor,
    final_energy: torch.Tensor
) -> Dict[str, float]:
    """
    Compute energy drift over trajectory.
    
    Args:
        initial_energy: Initial total energy (batch,)
        final_energy: Final total energy (batch,)
    
    Returns:
        Dictionary with drift statistics
    """
    drift = torch.abs(final_energy - initial_energy)
    relative_drift = drift / (initial_energy + 1e-8)
    
    return {
        'absolute_mean': drift.mean().item(),
        'absolute_std': drift.std().item(),
        'relative_mean': relative_drift.mean().item(),
        'relative_std': relative_drift.std().item()
    }


def compute_complementarity_violation(
    foot_positions: torch.Tensor,
    contact_forces: List[torch.Tensor],
    ground_height: float = 0.0
) -> float:
    """
    Compute complementarity condition violation.
    
    Complementarity: If foot above ground, force should be zero.
    
    Args:
        foot_positions: Foot positions (batch, 4, 3)
        contact_forces: List of forces for each foot
        ground_height: Ground plane height
    
    Returns:
        Mean violation
    """
    violations = []
    
    z_positions = foot_positions[:, :, 2]  # (batch, 4)
    clearance = z_positions - ground_height
    
    for i, forces in enumerate(contact_forces):
        foot_clearance = clearance[:, i]  # (batch,)
        normal_force = forces[:, 2]  # (batch,)
        
        # Violation when foot above ground but force > 0
        violation = torch.relu(foot_clearance) * torch.abs(normal_force)
        violations.append(violation)
    
    all_violations = torch.stack(violations).mean()
    return all_violations.item()


def compute_trajectory_metrics(
    predicted_positions: np.ndarray,
    target_positions: np.ndarray,
    predicted_velocities: np.ndarray,
    target_velocities: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive trajectory metrics.
    
    Args:
        predicted_positions: Predicted trajectory (timesteps, 3)
        target_positions: Target trajectory (timesteps, 3)
        predicted_velocities: Predicted velocities (timesteps, 3)
        target_velocities: Target velocities (timesteps, 3)
    
    Returns:
        Dictionary of metrics
    """
    # Convert to torch
    pred_pos = torch.from_numpy(predicted_positions).float()
    tgt_pos = torch.from_numpy(target_positions).float()
    pred_vel = torch.from_numpy(predicted_velocities).float()
    tgt_vel = torch.from_numpy(target_velocities).float()
    
    # Position errors
    pos_errors = torch.norm(pred_pos - tgt_pos, dim=-1)
    
    # Velocity errors
    vel_errors = torch.norm(pred_vel - tgt_vel, dim=-1)
    
    # Acceleration (from velocity)
    pred_acc = pred_vel[1:] - pred_vel[:-1]
    tgt_acc = tgt_vel[1:] - tgt_vel[:-1]
    acc_errors = torch.norm(pred_acc - tgt_acc, dim=-1)
    
    return {
        'position_rmse': torch.sqrt((pos_errors**2).mean()).item() * 1000,  # mm
        'position_max_error': pos_errors.max().item() * 1000,  # mm
        'velocity_rmse': torch.sqrt((vel_errors**2).mean()).item(),
        'velocity_max_error': vel_errors.max().item(),
        'acceleration_rmse': torch.sqrt((acc_errors**2).mean()).item(),
    }


def aggregate_metrics(
    metrics_list: List[Dict[str, float]]
) -> Dict[str, Tuple[float, float]]:
    """
    Aggregate metrics across multiple runs.
    
    Args:
        metrics_list: List of metric dictionaries
    
    Returns:
        Dictionary with (mean, std) for each metric
    """
    if not metrics_list:
        return {}
    
    aggregated = {}
    keys = metrics_list[0].keys()
    
    for key in keys:
        values = [m[key] for m in metrics_list]
        aggregated[key] = (np.mean(values), np.std(values))
    
    return aggregated


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    Format metrics dictionary as string.
    
    Args:
        metrics: Metrics dictionary
        precision: Decimal precision
    
    Returns:
        Formatted string
    """
    lines = []
    for key, value in metrics.items():
        if isinstance(value, tuple):  # (mean, std)
            lines.append(f"{key}: {value[0]:.{precision}f} ± {value[1]:.{precision}f}")
        else:
            lines.append(f"{key}: {value:.{precision}f}")
    return '\n'.join(lines)


# Example usage
if __name__ == "__main__":
    # Test metrics
    batch_size = 10
    
    # Position RMSE
    pred_pos = torch.randn(batch_size, 3)
    tgt_pos = torch.randn(batch_size, 3)
    rmse = compute_position_rmse(pred_pos, tgt_pos)
    print(f"Position RMSE: {rmse:.2f} mm")
    
    # Friction cone violation
    forces = [torch.randn(batch_size, 3) for _ in range(4)]
    violations = compute_friction_cone_violation(forces)
    print(f"Cone violations: {violations}")
    
    # Penetration
    foot_pos = torch.rand(batch_size, 4, 3) * 0.1  # Small heights
    penetration = compute_penetration_depth(foot_pos)
    print(f"Penetration: {penetration}")
