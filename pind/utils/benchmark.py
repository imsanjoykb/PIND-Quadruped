import torch
import time
import psutil
import os
from typing import Dict, Callable
from contextlib import contextmanager


class PerformanceBenchmark:
    """Benchmark model performance."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize benchmark."""
        self.device = device
        self.results = {}
    
    @contextmanager
    def timer(self, name: str):
        """Context manager for timing operations."""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.results[name] = elapsed
    
    def measure_inference_speed(
        self,
        model: torch.nn.Module,
        input_shape: tuple,
        num_iterations: int = 100,
        warmup: int = 10
    ) -> Dict[str, float]:
        """
        Measure inference speed.
        
        Args:
            model: Model to benchmark
            input_shape: Shape of input tensor
            num_iterations: Number of iterations
            warmup: Warmup iterations
        
        Returns:
            Dictionary with timing statistics
        """
        model.eval()
        model.to(self.device)
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape, device=self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        # Measure
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = model(dummy_input)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                times.append(elapsed)
        
        return {
            'mean_ms': torch.tensor(times).mean().item() * 1000,
            'std_ms': torch.tensor(times).std().item() * 1000,
            'min_ms': min(times) * 1000,
            'max_ms': max(times) * 1000,
            'fps': 1.0 / torch.tensor(times).mean().item()
        }
    
    def measure_memory_usage(
        self,
        model: torch.nn.Module,
        input_shape: tuple
    ) -> Dict[str, float]:
        """
        Measure memory usage.
        
        Args:
            model: Model to benchmark
            input_shape: Input shape
        
        Returns:
            Memory statistics in MB
        """
        model.to(self.device)
        
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            # Model parameters
            param_memory = sum(
                p.numel() * p.element_size() for p in model.parameters()
            ) / (1024 ** 2)
            
            # Forward pass
            dummy_input = torch.randn(*input_shape, device=self.device)
            _ = model(dummy_input)
            
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
            return {
                'param_memory_mb': param_memory,
                'peak_memory_mb': peak_memory,
                'total_memory_mb': peak_memory + param_memory
            }
        else:
            # CPU memory
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / (1024 ** 2)
            
            dummy_input = torch.randn(*input_shape)
            _ = model(dummy_input)
            
            mem_after = process.memory_info().rss / (1024 ** 2)
            
            param_memory = sum(
                p.numel() * p.element_size() for p in model.parameters()
            ) / (1024 ** 2)
            
            return {
                'param_memory_mb': param_memory,
                'runtime_memory_mb': mem_after - mem_before,
                'total_memory_mb': mem_after - mem_before + param_memory
            }
    
    def measure_flops(
        self,
        model: torch.nn.Module,
        input_shape: tuple
    ) -> Dict[str, int]:
        """
        Estimate FLOPs (rough estimate).
        
        Args:
            model: Model to analyze
            input_shape: Input shape
        
        Returns:
            FLOP estimates
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Rough estimate: 2 FLOPs per parameter per forward pass
        estimated_flops = total_params * 2
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'estimated_flops': estimated_flops,
            'estimated_gflops': estimated_flops / 1e9
        }
    
    def compare_models(
        self,
        models: Dict[str, torch.nn.Module],
        input_shape: tuple,
        num_iterations: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models.
        
        Args:
            models: Dictionary of model name -> model
            input_shape: Input shape
            num_iterations: Number of iterations
        
        Returns:
            Comparison results
        """
        results = {}
        
        for name, model in models.items():
            print(f"Benchmarking {name}...")
            
            speed = self.measure_inference_speed(model, input_shape, num_iterations)
            memory = self.measure_memory_usage(model, input_shape)
            flops = self.measure_flops(model, input_shape)
            
            results[name] = {
                **speed,
                **memory,
                **flops
            }
        
        return results


def profile_training_step(
    model: torch.nn.Module,
    batch_data: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    num_steps: int = 10
) -> Dict[str, float]:
    """
    Profile training step performance.
    
    Args:
        model: Model to train
        batch_data: Batch of data
        optimizer: Optimizer
        num_steps: Number of steps to profile
    
    Returns:
        Timing breakdown
    """
    forward_times = []
    backward_times = []
    optimizer_times = []
    
    for _ in range(num_steps):
        # Forward
        start = time.perf_counter()
        outputs = model(**batch_data)
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs
        forward_time = time.perf_counter() - start
        forward_times.append(forward_time)
        
        # Backward
        start = time.perf_counter()
        loss.backward()
        backward_time = time.perf_counter() - start
        backward_times.append(backward_time)
        
        # Optimizer
        start = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad()
        optimizer_time = time.perf_counter() - start
        optimizer_times.append(optimizer_time)
    
    return {
        'forward_mean_ms': torch.tensor(forward_times).mean().item() * 1000,
        'backward_mean_ms': torch.tensor(backward_times).mean().item() * 1000,
        'optimizer_mean_ms': torch.tensor(optimizer_times).mean().item() * 1000,
        'total_mean_ms': torch.tensor([
            f + b + o for f, b, o in zip(forward_times, backward_times, optimizer_times)
        ]).mean().item() * 1000
    }


# Example usage
if __name__ == "__main__":
    from pind.models import PIND
    
    # Create model
    model = PIND(nv=18, nu=12, hidden_dim=128)
    
    # Benchmark
    benchmark = PerformanceBenchmark()
    
    # Note: This is simplified - actual PIND forward requires specific inputs
    print("Performance benchmark would go here")
    print("(Requires proper input structure for PIND model)")
