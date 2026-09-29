"""
Physics-Informed Neural Dynamics (PIND) Model.

Complete PyTorch implementation with strict underactuation projections
and composite physics losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict


class ActuatorResidualMLP(nn.Module):
    """
    Models unmodeled drivetrain effects.
    
    Learns residual corrections for:
    - Drive damping
    - Motor back-EMF
    - Transmission friction
    
    Args:
        act_dim: Number of actuated joints (default: 12)
        hidden_dim: Hidden layer dimension (default: 128)
    """
    
    def __init__(self, act_dim: int = 12, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(act_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, act_dim)
        )
        # Initialize near zero so nominal physics dominates initially
        nn.init.uniform_(self.net[-1].weight, -1e-4, 1e-4)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, q_dot_j: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        Compute actuator residual.
        
        Args:
            q_dot_j: Joint velocities (batch, act_dim)
            tau: Applied torques (batch, act_dim)
        
        Returns:
            delta_tau: Residual torques (batch, act_dim)
        """
        inp = torch.cat([q_dot_j, tau], dim=-1)
        return self.net(inp)


class ContactComplianceNetwork(nn.Module):
    """
    Predicts compliant contact wrenches for each foot.
    
    Enforces:
    - Smooth contact activation
    - Unilateral normal force constraint (F_z >= 0)
    - Physical plausibility through gating
    
    Args:
        hidden_dim: Hidden layer dimension (default: 128)
        eps_c: Contact activation threshold (default: 0.002)
    """
    
    def __init__(self, hidden_dim: int = 128, eps_c: float = 0.002):
        super().__init__()
        self.eps_c = eps_c
        self.net = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(
        self, p_foot: torch.Tensor, v_foot: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict contact wrench for a single foot.
        
        Args:
            p_foot: Foot position [x, y, z] (batch, 3)
            v_foot: Foot velocity [\dot{x}, \dot{y}, \dot{z}] (batch, 3)
        
        Returns:
            wrench: Contact force [F_x, F_y, F_z] (batch, 3)
        """
        z_clearance = p_foot[:, 2:3]
        # Smooth gate: approaching contact enables force
        gate = torch.sigmoid(-z_clearance / self.eps_c)
        
        inp = torch.cat([p_foot, v_foot], dim=-1)
        raw_force = self.net(inp)
        
        # Enforce unilateral normal force constraint: F_z >= 0
        fz = F.softplus(raw_force[:, 2:3])
        fxy = raw_force[:, 0:2]
        
        wrench = torch.cat([fxy, fz], dim=-1)
        return gate * wrench


class PIND(nn.Module):
    """
    Physics-Informed Neural Dynamics (PIND) model.
    
    Hybrid framework combining analytical rigid-body dynamics with learned
    residual corrections for actuator and contact effects.
    
    Args:
        nv: Number of velocity DOFs (default: 18)
        nu: Number of actuated joints (default: 12)
        hidden_dim: Hidden layer dimension (default: 128)
        eps_c: Contact activation threshold (default: 0.002)
        mu: Friction coefficient (default: 0.8)
    """
    
    def __init__(
        self,
        nv: int = 18,
        nu: int = 12,
        hidden_dim: int = 128,
        eps_c: float = 0.002,
        mu: float = 0.8
    ):
        super().__init__()
        self.nv = nv
        self.nu = nu
        self.mu = mu
        
        # B selection matrix: [0_{12x6}, I_{12x12}]^T
        B = torch.zeros(nv, nu)
        B[6:, :] = torch.eye(nu)
        self.register_buffer("B", B)

        # Neural Sub-modules
        self.actuator_net = ActuatorResidualMLP(act_dim=nu, hidden_dim=hidden_dim)
        self.contact_net = ContactComplianceNetwork(hidden_dim=hidden_dim, eps_c=eps_c)

    def forward_acceleration(
        self,
        q: torch.Tensor,
        v: torch.Tensor,
        u: torch.Tensor,
        M_nom: torch.Tensor,
        h_nom: torch.Tensor,
        p_feet: torch.Tensor,
        J_feet: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Compute generalized acceleration using Eq. (8) from the paper.
        
        M(q) * \dot{\nu} = u_act + δ_act + J_c^T * λ - h(q, v)
        
        Args:
            q: Generalized positions (batch, nq)
            v: Generalized velocities (batch, nv)
            u: Control inputs (batch, nu)
            M_nom: Mass matrix (batch, nv, nv)
            h_nom: Nonlinear bias forces (batch, nv)
            p_feet: Foot positions (batch, 4, 3)
            J_feet: Foot Jacobians (batch, 4, 3, nv)
        
        Returns:
            v_dot: Generalized accelerations (batch, nv)
            predicted_lambdas: Contact forces for each foot
            delta_tau: Actuator residuals (batch, nu)
        """
        batch_size = q.shape[0]
        q_dot_j = v[:, 6:]  # joint velocities
        
        # 1. Projected internal drivetrain residual
        delta_tau = self.actuator_net(q_dot_j, u)  # (batch, 12)
        generalized_internal = torch.matmul(delta_tau, self.B.T)  # (batch, 18)

        # 2. Foot contact forces and mapping
        total_contact_wrench = torch.zeros(batch_size, self.nv, device=q.device)
        predicted_lambdas = []
        
        for leg_idx in range(4):
            p_leg = p_feet[:, leg_idx, :]  # (batch, 3)
            J_leg = J_feet[:, leg_idx, :, :]  # (batch, 3, 18)
            
            # v_foot = J * v
            v_leg = torch.bmm(J_leg, v.unsqueeze(-1)).squeeze(-1)  # (batch, 3)
            
            lam = self.contact_net(p_leg, v_leg)  # (batch, 3)
            predicted_lambdas.append(lam)
            
            # J^T * lambda
            wrench_gen = torch.bmm(J_leg.transpose(1, 2), lam.unsqueeze(-1)).squeeze(-1)
            total_contact_wrench = total_contact_wrench + wrench_gen

        # 3. Formulate equations of motion and solve for acceleration
        u_act = torch.matmul(u, self.B.T)
        rhs = u_act + generalized_internal + total_contact_wrench - h_nom
        
        # Solve M(q) * \dot{\nu} = rhs via Cholesky decomposition
        L = torch.linalg.cholesky(M_nom)
        v_dot = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)

        return v_dot, predicted_lambdas, delta_tau

    def compute_composite_loss(
        self,
        v_dot_pred: torch.Tensor,
        v_dot_target: torch.Tensor,
        predicted_lambdas: List[torch.Tensor],
        p_feet: torch.Tensor,
        M_nom: torch.Tensor,
        h_nom: torch.Tensor,
        u: torch.Tensor,
        delta_tau: torch.Tensor,
        J_feet: torch.Tensor,
        beta_dyn: float = 1.0,
        beta_cone: float = 10.0,
        beta_pen: float = 10.0,
        alpha_comp: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute composite physics-informed loss.
        
        L_total = L_data + β_dyn*L_dyn + β_cone*L_cone + β_pen*L_pen
        
        Args:
            v_dot_pred: Predicted accelerations (batch, nv)
            v_dot_target: Ground truth accelerations (batch, nv)
            predicted_lambdas: Contact forces for each foot
            p_feet: Foot positions (batch, 4, 3)
            M_nom: Mass matrix (batch, nv, nv)
            h_nom: Nonlinear bias forces (batch, nv)
            u: Control inputs (batch, nu)
            delta_tau: Actuator residuals (batch, nu)
            J_feet: Foot Jacobians (batch, 4, 3, nv)
            beta_dyn: Weight for dynamics residual loss
            beta_cone: Weight for friction cone loss
            beta_pen: Weight for penetration/complementarity loss
            alpha_comp: Weight for complementarity within penetration loss
        
        Returns:
            total_loss: Weighted sum of all losses
            metrics: Dictionary of individual loss components
        """
        # 1. Supervised Data Loss
        l_data = F.mse_loss(v_dot_pred, v_dot_target)

        # 2. Generalized Newton-Euler Momentum Residual Loss
        u_act = torch.matmul(u, self.B.T)
        gen_int = torch.matmul(delta_tau, self.B.T)
        
        wrench_sum = torch.zeros_like(gen_int)
        for i, lam in enumerate(predicted_lambdas):
            J_i = J_feet[:, i, :, :]
            wrench_sum = wrench_sum + torch.bmm(J_i.transpose(1, 2), lam.unsqueeze(-1)).squeeze(-1)

        r_dyn = torch.bmm(M_nom, v_dot_pred.unsqueeze(-1)).squeeze(-1) + h_nom - u_act - gen_int - wrench_sum
        l_dyn = torch.mean(torch.sum(r_dyn ** 2, dim=-1))

        # 3. Friction Cone Admissibility Loss
        l_cone = torch.tensor(0.0, device=v_dot_pred.device)
        for lam in predicted_lambdas:
            fx, fy, fz = lam[:, 0], lam[:, 1], lam[:, 2]
            f_tan = torch.sqrt(fx ** 2 + fy ** 2 + 1e-8)
            cone_violation = F.relu(f_tan - self.mu * fz)
            l_cone = l_cone + torch.mean(cone_violation ** 2)

        # 4. Kinematic Non-Penetration and Contact Complementarity
        l_pen = torch.tensor(0.0, device=v_dot_pred.device)
        for i, lam in enumerate(predicted_lambdas):
            z_foot = p_feet[:, i, 2]
            penetration = F.relu(-z_foot)
            complementarity = torch.abs(F.relu(z_foot) * lam[:, 2])
            l_pen = l_pen + torch.mean(penetration ** 2 + alpha_comp * complementarity)

        total_loss = l_data + beta_dyn * l_dyn + beta_cone * l_cone + beta_pen * l_pen
        
        return total_loss, {
            "l_data": l_data.item(),
            "l_dyn": l_dyn.item(),
            "l_cone": l_cone.item(),
            "l_pen": l_pen.item()
        }
