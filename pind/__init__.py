"""PIND: Physics-Informed Neural Dynamics for Quadruped Robots."""

__version__ = "0.1.0"
__author__ = "Sanjoy Kumar Barman"
__email__ = "[YOUR_EMAIL]@example.com"  # Update with your actual email

from pind.models.pind import PIND
from pind.models.baselines import BlackBoxMLP, NeuralODEVectorField, UnconstrainedResidualMLP

__all__ = [
    "PIND",
    "BlackBoxMLP",
    "NeuralODEVectorField",
    "UnconstrainedResidualMLP",
]
