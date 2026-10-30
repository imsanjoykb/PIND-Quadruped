"""
Configuration management utilities.

Load and validate YAML configurations.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import copy


class ConfigValidator:
    """Validate configuration dictionaries."""
    
    REQUIRED_KEYS = {
        'model': ['name', 'nv', 'nu'],
        'training': ['epochs', 'batch_size', 'learning_rate'],
        'system': ['device', 'seed']
    }
    
    @staticmethod
    def validate(config: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate configuration.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            (is_valid, error_message)
        """
        for section, required in ConfigValidator.REQUIRED_KEYS.items():
            if section not in config:
                return False, f"Missing section: {section}"
            
            for key in required:
                if key not in config[section]:
                    return False, f"Missing key: {section}.{key}"
        
        # Type checks
        if not isinstance(config['training']['epochs'], int):
            return False, "epochs must be integer"
        
        if config['training']['epochs'] <= 0:
            return False, "epochs must be positive"
        
        if not isinstance(config['training']['batch_size'], int):
            return False, "batch_size must be integer"
        
        if config['training']['batch_size'] <= 0:
            return False, "batch_size must be positive"
        
        return True, ""


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML file
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate
    is_valid, error_msg = ConfigValidator.validate(config)
    if not is_valid:
        raise ValueError(f"Invalid config: {error_msg}")
    
    return config


def save_config(config: Dict[str, Any], output_path: Path):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Output path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations, with override taking precedence.
    
    Args:
        base: Base configuration
        override: Override configuration
    
    Returns:
        Merged configuration
    """
    merged = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def update_config_from_args(config: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """
    Update configuration from command-line arguments.
    
    Args:
        config: Base configuration
        args: Argparse arguments
    
    Returns:
        Updated configuration
    """
    config = copy.deepcopy(config)
    
    # Update from args if provided
    if hasattr(args, 'epochs') and args.epochs is not None:
        config['training']['epochs'] = args.epochs
    
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    
    if hasattr(args, 'learning_rate') and args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    
    if hasattr(args, 'device') and args.device is not None:
        config['system']['device'] = args.device
    
    if hasattr(args, 'seed') and args.seed is not None:
        config['system']['seed'] = args.seed
    
    return config


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'model': {
            'name': 'pind',
            'nv': 18,
            'nu': 12,
            'hidden_dim': 128,
            'eps_c': 0.002,
            'friction_coeff': 0.8
        },
        'training': {
            'epochs': 30,
            'batch_size': 64,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'grad_clip': 5.0,
            'optimizer': 'adamw'
        },
        'loss_weights': {
            'beta_dyn': 1.0,
            'beta_cone': 10.0,
            'beta_pen': 10.0,
            'alpha_comp': 1.0
        },
        'data': {
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1
        },
        'logging': {
            'log_interval': 10,
            'checkpoint_interval': 5,
            'use_tensorboard': True
        },
        'system': {
            'device': 'cuda',
            'num_workers': 4,
            'pin_memory': True,
            'seed': 42
        }
    }


# Example usage
if __name__ == "__main__":
    # Test config loading
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.yaml'
        
        # Save default config
        config = get_default_config()
        save_config(config, config_path)
        print(f"Saved config to {config_path}")
        
        # Load it back
        loaded = load_config(config_path)
        print("Loaded config:")
        print(yaml.dump(loaded, default_flow_style=False))
        
        # Test validation
        is_valid, msg = ConfigValidator.validate(loaded)
        print(f"Valid: {is_valid}")
