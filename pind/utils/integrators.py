import torch
from typing import Callable, Any


def rk4_step(
    vector_field: Callable,
    x: torch.Tensor,
    u: torch.Tensor,
    dt: float,
    **kwargs: Any
) -> torch.Tensor:
    """
    Computes an explicit 4th-order Runge-Kutta step: x_{k+1} = RK4(f, x_k, u_k, dt).
    
    Args:
        vector_field: Callable f(x, u, **kwargs) returning \dot{x}.
        x: Tensor of shape (batch_size, state_dim).
        u: Tensor of shape (batch_size, control_dim).
        dt: Float, integration time step.
        **kwargs: Additional keyword arguments passed to vector_field.
    
    Returns:
        x_next: Tensor of shape (batch_size, state_dim).
    
    Example:
        >>> def dynamics(x, u):
        ...     return -x + u  # Simple linear dynamics
        >>> x0 = torch.tensor([[1.0, 2.0]])
        >>> u0 = torch.tensor([[0.5, 0.5]])
        >>> x1 = rk4_step(dynamics, x0, u0, dt=0.01)
    """
    k1 = vector_field(x, u, **kwargs)
    k2 = vector_field(x + 0.5 * dt * k1, u, **kwargs)
    k3 = vector_field(x + 0.5 * dt * k2, u, **kwargs)
    k4 = vector_field(x + dt * k3, u, **kwargs)
    
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x_next
