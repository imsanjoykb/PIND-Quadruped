"""
Trajectory data generation using MuJoCo simulation.

Generates genuine interaction trajectories and extracts analytical
nominal physics tensors required for PIND training.
"""

import numpy as np
import torch
import os
import mujoco
from torch.utils.data import Dataset
from pind.utils.mujoco_interface import QuadrupedMuJoCoInterface
from tqdm import tqdm
from typing import Dict, Optional
import argparse


class QuadrupedDataset(Dataset):
    """
    PyTorch Dataset for quadruped dynamics data.
    
    Each sample contains:
    - q, v, u: Current state and control
    - q_next, v_next: Next state
    - v_dot_gt: Ground truth acceleration
    - M_nom, h_nom: Nominal physics matrices
    - p_feet, J_feet: Foot positions and Jacobians
    """
    
    def __init__(self, data_path: str):
        """
        Load dataset from file.
        
        Args:
            data_path: Path to .pt file containing trajectory data
        """
        self.data = torch.load(data_path)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


def collect_trajectories(
    num_trajectories: int = 20,
    trajectory_length: int = 500,
    dt: float = 0.02,
    save_path: str = "data/go1_dataset.pt",
    xml_path: Optional[str] = None,
    seed: Optional[int] = None
) -> None:
    """
    Generate training data from MuJoCo simulation.
    
    Simulates quadruped robot with sinusoidal trotting gait and collects
    state transitions along with analytical physics quantities.
    
    Args:
        num_trajectories: Number of trajectories to collect
        trajectory_length: Number of timesteps per trajectory
        dt: Simulation timestep (seconds)
        save_path: Path to save dataset
        xml_path: Optional path to custom MJCF model
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sim = QuadrupedMuJoCoInterface(xml_path=xml_path)
    
    dataset = []
    print(f"Generating {num_trajectories} trajectories ({trajectory_length * dt}s each)...")

    for traj_idx in tqdm(range(num_trajectories), desc="Trajectories"):
        # Reset to standing initial condition with small noise
        mujoco.mj_resetData(sim.model, sim.data)
        sim.data.qpos[2] = 0.30 + np.random.uniform(-0.02, 0.02)
        sim.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # identity quaternion
        
        # Default nominal joint angles (standing pose)
        sim.data.qpos[7:] = [
            0.0, 0.7, -1.4,   # FL: hip_x, hip_y, knee
            0.0, 0.7, -1.4,   # FR
            0.0, 0.7, -1.4,   # RL
            0.0, 0.7, -1.4    # RR
        ]
        mujoco.mj_forward(sim.model, sim.data)

        for step in range(trajectory_length):
            q_k = sim.data.qpos.copy()
            v_k = sim.data.qvel.copy()
            
            # Analytical Nominal Physics at current state
            M_nom, h_nom, p_feet, J_feet = sim.compute_nominal_dynamics(q_k, v_k)

            # Generate sinusoidal trotting control input with exploration noise
            t = step * dt
            # Trotting gait: diagonal pairs move together (FL+RR, FR+RL)
            u_nominal = 3.0 * np.sin(
                2.0 * np.pi * 1.5 * t + np.array([0, np.pi, np.pi, 0] * 3)
            )
            noise = np.random.normal(0, 1.2, size=sim.nu)
            u_k = np.clip(u_nominal + noise, -15.0, 15.0)

            # Step the true simulator forward at 1 kHz internal physics
            sim.data.ctrl[:] = u_k
            substeps = int(dt / sim.model.opt.timestep)
            for _ in range(substeps):
                mujoco.mj_step(sim.model, sim.data)

            q_next = sim.data.qpos.copy()
            v_next = sim.data.qvel.copy()
            
            # Ground-truth numerical generalized acceleration
            v_dot_gt = (v_next - v_k) / dt

            dataset.append({
                "q": torch.tensor(q_k, dtype=torch.float32),
                "v": torch.tensor(v_k, dtype=torch.float32),
                "u": torch.tensor(u_k, dtype=torch.float32),
                "q_next": torch.tensor(q_next, dtype=torch.float32),
                "v_next": torch.tensor(v_next, dtype=torch.float32),
                "v_dot_gt": torch.tensor(v_dot_gt, dtype=torch.float32),
                "M_nom": torch.tensor(M_nom, dtype=torch.float32),
                "h_nom": torch.tensor(h_nom, dtype=torch.float32),
                "p_feet": torch.tensor(p_feet, dtype=torch.float32),
                "J_feet": torch.tensor(J_feet, dtype=torch.float32),
            })

    torch.save(dataset, save_path)
    print(f"✓ Dataset with {len(dataset)} transitions saved to {save_path}")
    print(f"  Total duration: {len(dataset) * dt:.1f}s")
    print(f"  Data size: {os.path.getsize(save_path) / 1024 / 1024:.1f} MB")


def main():
    """Command-line interface for data generation."""
    parser = argparse.ArgumentParser(
        description="Generate quadruped dynamics training data"
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=20,
        help="Number of trajectories to collect"
    )
    parser.add_argument(
        "--trajectory-length",
        type=int,
        default=500,
        help="Number of timesteps per trajectory"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="Simulation timestep (seconds)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/go1_dataset.pt",
        help="Output path for dataset"
    )
    parser.add_argument(
        "--xml-path",
        type=str,
        default=None,
        help="Optional path to custom MJCF model"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    collect_trajectories(
        num_trajectories=args.num_trajectories,
        trajectory_length=args.trajectory_length,
        dt=args.dt,
        save_path=args.output,
        xml_path=args.xml_path,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
