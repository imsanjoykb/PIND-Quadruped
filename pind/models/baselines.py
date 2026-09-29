"""
Baseline Models for Comparison.

Implements:
- BlackBoxMLP: Pure data-driven discrete dynamics
- NeuralODEVectorField: Continuous-time neural ODE
- UnconstrainedResidualMLP: Hybrid model without physics constraints
"""

import torch
import torch.nn as nn
from typing import Optional


class BlackBoxMLP(nn.Module):
    """
    Pure black-box MLP baseline.
    
    Direct discrete state-to-state mapping: Δx = MLP(x, u)
    No physics priors, purely data-driven.
    
    Args:
        state_dim: State dimension (default: 37 for 19 pos + 18 vel)
        act_dim: Action dimension (default: 12)
        hidden_dim: Hidden layer dimension (default: 256)
    """
    
    def __init__(
        self,
        state_dim: int = 37,
        act_dim: int = 12,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + act_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Predict state change.
        
        Args:
            x: Current state (batch, state_dim)
            u: Control input (batch, act_dim)
        
        Returns:
            delta_x: Predicted state change (batch, state_dim)
        """
        return self.net(torch.cat([x, u], dim=-1))


class NeuralODEVectorField(nn.Module):
    """
    Continuous-time Neural ODE baseline.
    
    Continuous vector field: ẋ = NN(x, u)
    Uses neural network to directly predict state derivatives.
    
    Args:
        state_dim: State dimension (default: 37)
        act_dim: Action dimension (default: 12)
        hidden_dim: Hidden layer dimension (default: 256)
    """
    
    def __init__(
        self,
        state_dim: int = 37,
        act_dim: int = 12,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + act_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute state derivative.
        
        Args:
            x: Current state (batch, state_dim)
            u: Control input (batch, act_dim)
        
        Returns:
            x_dot: State derivative (batch, state_dim)
        """
        return self.net(torch.cat([x, u], dim=-1))


class UnconstrainedResidualMLP(nn.Module):
    """
    Unconstrained residual physics baseline.
    
    Hybrid model lacking physical loss constraints:
    ν̇ = ν̇_nom + MLP(x, u)
    
    Combines nominal physics with learned residual but without
    the physics-informed loss terms used in PIND.
    
    Args:
        state_dim: State dimension (default: 37)
        act_dim: Action dimension (default: 12)
        nv: Velocity DOF dimension (default: 18)
        hidden_dim: Hidden layer dimension (default: 128)
    """
    
    def __init__(
        self,
        state_dim: int = 37,
        act_dim: int = 12,
        nv: int = 18,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + act_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, nv)
        )
        # Initialize near zero for stability
        nn.init.uniform_(self.net[-1].weight, -1e-4, 1e-4)

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        v_dot_nom: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict acceleration with nominal + residual.
        
        Args:
            x: Current state (batch, state_dim)
            u: Control input (batch, act_dim)
            v_dot_nom: Nominal acceleration from physics (batch, nv)
        
        Returns:
            v_dot: Predicted acceleration (batch, nv)
        """
        res = self.net(torch.cat([x, u], dim=-1))
        return v_dot_nom + res
