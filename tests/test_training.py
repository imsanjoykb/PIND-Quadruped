import pytest
import torch
import tempfile
import os
from pathlib import Path
from pind.models import PIND
from pind.data import QuadrupedDataset, collect_trajectories


class TestTrainingBasics:
    """Test basic training functionality."""
    
    @pytest.fixture
    def model(self):
        """Create PIND model."""
        return PIND(nv=18, nu=12, hidden_dim=32)  # Small for testing
    
    @pytest.fixture
    def sample_batch(self):
        """Create sample batch."""
        batch_size = 4
        return {
            'q': torch.randn(batch_size, 19),
            'v': torch.randn(batch_size, 18),
            'u': torch.randn(batch_size, 12),
            'v_dot_gt': torch.randn(batch_size, 18),
            'M_nom': torch.eye(18).unsqueeze(0).repeat(batch_size, 1, 1),
            'h_nom': torch.randn(batch_size, 18),
            'p_feet': torch.randn(batch_size, 4, 3),
            'J_feet': torch.randn(batch_size, 4, 3, 18),
        }
    
    def test_forward_pass(self, model, sample_batch):
        """Test forward pass works."""
        v_dot, lambdas, delta_tau = model.forward_acceleration(
            sample_batch['q'],
            sample_batch['v'],
            sample_batch['u'],
            sample_batch['M_nom'],
            sample_batch['h_nom'],
            sample_batch['p_feet'],
            sample_batch['J_feet']
        )
        
        assert v_dot.shape == (4, 18)
        assert len(lambdas) == 4
        assert delta_tau.shape == (4, 12)
    
    def test_loss_computation(self, model, sample_batch):
        """Test loss computation."""
        v_dot, lambdas, delta_tau = model.forward_acceleration(
            sample_batch['q'],
            sample_batch['v'],
            sample_batch['u'],
            sample_batch['M_nom'],
            sample_batch['h_nom'],
            sample_batch['p_feet'],
            sample_batch['J_feet']
        )
        
        loss, metrics = model.compute_composite_loss(
            v_dot, sample_batch['v_dot_gt'], lambdas,
            sample_batch['p_feet'], sample_batch['M_nom'],
            sample_batch['h_nom'], sample_batch['u'],
            delta_tau, sample_batch['J_feet']
        )
        
        assert loss.item() >= 0, "Loss should be non-negative"
        assert 'l_data' in metrics
        assert 'l_dyn' in metrics
        assert 'l_cone' in metrics
        assert 'l_pen' in metrics
    
    def test_backward_pass(self, model, sample_batch):
        """Test backward pass and gradient computation."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        v_dot, lambdas, delta_tau = model.forward_acceleration(
            sample_batch['q'],
            sample_batch['v'],
            sample_batch['u'],
            sample_batch['M_nom'],
            sample_batch['h_nom'],
            sample_batch['p_feet'],
            sample_batch['J_feet']
        )
        
        loss, _ = model.compute_composite_loss(
            v_dot, sample_batch['v_dot_gt'], lambdas,
            sample_batch['p_feet'], sample_batch['M_nom'],
            sample_batch['h_nom'], sample_batch['u'],
            delta_tau, sample_batch['J_feet']
        )
        
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Gradient should be computed"
        
        optimizer.step()
    
    def test_training_reduces_loss(self, model, sample_batch):
        """Test training actually reduces loss."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        
        # Initial loss
        v_dot, lambdas, delta_tau = model.forward_acceleration(
            sample_batch['q'], sample_batch['v'], sample_batch['u'],
            sample_batch['M_nom'], sample_batch['h_nom'],
            sample_batch['p_feet'], sample_batch['J_feet']
        )
        loss1, _ = model.compute_composite_loss(
            v_dot, sample_batch['v_dot_gt'], lambdas,
            sample_batch['p_feet'], sample_batch['M_nom'],
            sample_batch['h_nom'], sample_batch['u'],
            delta_tau, sample_batch['J_feet']
        )
        
        # Train for a few steps
        for _ in range(10):
            optimizer.zero_grad()
            v_dot, lambdas, delta_tau = model.forward_acceleration(
                sample_batch['q'], sample_batch['v'], sample_batch['u'],
                sample_batch['M_nom'], sample_batch['h_nom'],
                sample_batch['p_feet'], sample_batch['J_feet']
            )
            loss, _ = model.compute_composite_loss(
                v_dot, sample_batch['v_dot_gt'], lambdas,
                sample_batch['p_feet'], sample_batch['M_nom'],
                sample_batch['h_nom'], sample_batch['u'],
                delta_tau, sample_batch['J_feet']
            )
            loss.backward()
            optimizer.step()
        
        # Final loss
        v_dot, lambdas, delta_tau = model.forward_acceleration(
            sample_batch['q'], sample_batch['v'], sample_batch['u'],
            sample_batch['M_nom'], sample_batch['h_nom'],
            sample_batch['p_feet'], sample_batch['J_feet']
        )
        loss2, _ = model.compute_composite_loss(
            v_dot, sample_batch['v_dot_gt'], lambdas,
            sample_batch['p_feet'], sample_batch['M_nom'],
            sample_batch['h_nom'], sample_batch['u'],
            delta_tau, sample_batch['J_feet']
        )
        
        assert loss2.item() < loss1.item(), "Training should reduce loss"


class TestCheckpointing:
    """Test model checkpointing and loading."""
    
    @pytest.fixture
    def model(self):
        """Create model."""
        return PIND(nv=18, nu=12, hidden_dim=32)
    
    def test_save_load_checkpoint(self, model):
        """Test saving and loading checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "model.pt")
            
            # Save
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': 10,
            }, checkpoint_path)
            
            # Create new model and load
            new_model = PIND(nv=18, nu=12, hidden_dim=32)
            checkpoint = torch.load(checkpoint_path)
            new_model.load_state_dict(checkpoint['model_state_dict'])
            
            # Compare parameters
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                assert torch.allclose(p1, p2), "Parameters should match"
    
    def test_checkpoint_includes_optimizer(self, model):
        """Test checkpoint can include optimizer state."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': 5,
            }, checkpoint_path)
            
            checkpoint = torch.load(checkpoint_path)
            
            assert 'model_state_dict' in checkpoint
            assert 'optimizer_state_dict' in checkpoint
            assert checkpoint['epoch'] == 5


class TestGradientHandling:
    """Test gradient clipping and numerical stability."""
    
    @pytest.fixture
    def model(self):
        return PIND(nv=18, nu=12, hidden_dim=32)
    
    @pytest.fixture
    def sample_batch(self):
        batch_size = 2
        return {
            'q': torch.randn(batch_size, 19),
            'v': torch.randn(batch_size, 18),
            'u': torch.randn(batch_size, 12),
            'v_dot_gt': torch.randn(batch_size, 18),
            'M_nom': torch.eye(18).unsqueeze(0).repeat(batch_size, 1, 1),
            'h_nom': torch.randn(batch_size, 18),
            'p_feet': torch.randn(batch_size, 4, 3),
            'J_feet': torch.randn(batch_size, 4, 3, 18),
        }
    
    def test_gradient_clipping(self, model, sample_batch):
        """Test gradient clipping works."""
        optimizer = torch.optim.Adam(model.parameters())
        
        v_dot, lambdas, delta_tau = model.forward_acceleration(
            sample_batch['q'], sample_batch['v'], sample_batch['u'],
            sample_batch['M_nom'], sample_batch['h_nom'],
            sample_batch['p_feet'], sample_batch['J_feet']
        )
        loss, _ = model.compute_composite_loss(
            v_dot, sample_batch['v_dot_gt'], lambdas,
            sample_batch['p_feet'], sample_batch['M_nom'],
            sample_batch['h_nom'], sample_batch['u'],
            delta_tau, sample_batch['J_feet']
        )
        
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        # Check gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        assert total_norm <= max_norm * 1.1, "Gradients should be clipped"
    
    def test_no_nan_gradients(self, model, sample_batch):
        """Test gradients don't become NaN."""
        optimizer = torch.optim.Adam(model.parameters())
        
        v_dot, lambdas, delta_tau = model.forward_acceleration(
            sample_batch['q'], sample_batch['v'], sample_batch['u'],
            sample_batch['M_nom'], sample_batch['h_nom'],
            sample_batch['p_feet'], sample_batch['J_feet']
        )
        loss, _ = model.compute_composite_loss(
            v_dot, sample_batch['v_dot_gt'], lambdas,
            sample_batch['p_feet'], sample_batch['M_nom'],
            sample_batch['h_nom'], sample_batch['u'],
            delta_tau, sample_batch['J_feet']
        )
        
        optimizer.zero_grad()
        loss.backward()
        
        for p in model.parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), "Gradients should not be NaN"
                assert not torch.isinf(p.grad).any(), "Gradients should not be inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
