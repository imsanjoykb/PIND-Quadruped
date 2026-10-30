"""Utility functions and classes for PIND."""

from pind.utils.integrators import rk4_step
from pind.utils.mujoco_interface import QuadrupedMuJoCoInterface
from pind.utils.logging import setup_logger, MetricsLogger, TrainingLogger
from pind.utils.config import load_config, save_config, get_default_config
from pind.utils.metrics import (
    compute_position_rmse,
    compute_velocity_rmse,
    compute_friction_cone_violation,
    compute_penetration_depth,
    compute_trajectory_metrics,
    aggregate_metrics,
    format_metrics
)
from pind.utils.benchmark import PerformanceBenchmark

__all__ = [
    # Integrators
    "rk4_step",
    # MuJoCo
    "QuadrupedMuJoCoInterface",
    # Logging
    "setup_logger",
    "MetricsLogger",
    "TrainingLogger",
    # Config
    "load_config",
    "save_config",
    "get_default_config",
    # Metrics
    "compute_position_rmse",
    "compute_velocity_rmse",
    "compute_friction_cone_violation",
    "compute_penetration_depth",
    "compute_trajectory_metrics",
    "aggregate_metrics",
    "format_metrics",
    # Benchmark
    "PerformanceBenchmark",
]
