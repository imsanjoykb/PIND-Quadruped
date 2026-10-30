import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path


@pytest.fixture(scope="session")
def tmp_data_dir():
    """Create temporary directory for test data."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def device():
    """Get device for testing (CPU by default)."""
    return torch.device('cpu')


@pytest.fixture
def random_seed():
    """Set random seeds for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed


@pytest.fixture
def small_model():
    """Create small model for fast testing."""
    from pind.models import PIND
    return PIND(nv=18, nu=12, hidden_dim=16)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'model': {
            'name': 'pind',
            'nv': 18,
            'nu': 12,
            'hidden_dim': 32,
            'eps_c': 0.002,
            'friction_coeff': 0.8
        },
        'training': {
            'epochs': 2,
            'batch_size': 4,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'grad_clip': 5.0
        },
        'loss_weights': {
            'beta_dyn': 1.0,
            'beta_cone': 10.0,
            'beta_pen': 10.0,
            'alpha_comp': 1.0
        },
        'system': {
            'device': 'cpu',
            'num_workers': 0,
            'seed': 42
        }
    }


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


# Skip GPU tests if CUDA not available
def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if CUDA not available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
