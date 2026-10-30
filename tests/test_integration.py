"""
Integration tests for complete workflows.

Tests end-to-end pipelines from data generation to evaluation.
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path
from pind.data import collect_trajectories, QuadrupedDataset
from pind.models import PIND
from torch.utils.data import DataLoader


@pytest.mark.slow
class TestEndToEndWorkflow:
    """Test complete workflow from data to training to evaluation."""
    
    @pytest.fixture(scope="class")
    def workspace(self):
        """Create temporary workspace."""
        tmpdir = tempfile.mkdtemp()
        yield Path(tmpdir)
        import shutil
        shutil.rmtree(tmpdir)
    
    def test_data_generation(self, workspace):
        """Test data generation."""
        data_path = workspace / "dataset.pt"
        
        collect_trajectories(
            num_trajectories=2,
            trajectory_length=20,
            save_path=str(data_path),
            seed=42
        )
        
        assert data_path.exists()
        dataset = QuadrupedDataset(str(data_path))
        assert len(dataset) == 40  # 2 * 20
    
    def test_model_training(self, workspace):
        """Test model can be trained."""
        # Use dataset from previous test
        data_path = workspace / "dataset.pt"
        
        if not data_path.exists():
            collect_trajectories(2, 20, save_path=str(data_path), seed=42)
        
        dataset = QuadrupedDataset(str(data_path))
        
        def collate_fn(batch):
            return {k: torch.stack([b[k] for b in batch]) for k in batch[0].keys()}
        
        loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
        
        # Train for a few steps
        model = PIND(nv=18, nu=12, hidden_dim=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        initial_loss = None
        for i, batch in enumerate(loader):
            if i >= 3:  # Just 3 batches
                break
            
            v_dot, lambdas, delta_tau = model.forward_acceleration(
                batch['q'], batch['v'], batch['u'],
                batch['M_nom'], batch['h_nom'],
                batch['p_feet'], batch['J_feet']
            )
            
            loss, _ = model.compute_composite_loss(
                v_dot, batch['v_dot_gt'], lambdas,
                batch['p_feet'], batch['M_nom'],
                batch['h_nom'], batch['u'],
                delta_tau, batch['J_feet']
            )
            
            if initial_loss is None:
                initial_loss = loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        assert loss.item() <= initial_loss, "Loss should decrease"
    
    def test_model_save_load(self, workspace):
        """Test saving and loading trained model."""
        model_path = workspace / "model.pt"
        
        # Create and save model
        model = PIND(nv=18, nu=12, hidden_dim=16)
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {'hidden_dim': 16}
        }, model_path)
        
        # Load model
        checkpoint = torch.load(model_path)
        new_model = PIND(nv=18, nu=12, hidden_dim=16)
        new_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Verify parameters match
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)


@pytest.mark.slow
class TestMultiSeedExperiment:
    """Test multi-seed experiment reproducibility."""
    
    def test_different_seeds_give_different_results(self):
        """Test different seeds produce different results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate data with seed 1
            path1 = os.path.join(tmpdir, "data1.pt")
            collect_trajectories(1, 10, save_path=path1, seed=1)
            
            # Generate data with seed 2
            path2 = os.path.join(tmpdir, "data2.pt")
            collect_trajectories(1, 10, save_path=path2, seed=2)
            
            data1 = torch.load(path1)
            data2 = torch.load(path2)
            
            # Should be different
            assert not torch.allclose(data1[0]['q'], data2[0]['q'])
    
    def test_same_seed_gives_same_results(self):
        """Test same seed produces identical results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate twice with same seed
            path1 = os.path.join(tmpdir, "data1.pt")
            collect_trajectories(1, 10, save_path=path1, seed=42)
            
            path2 = os.path.join(tmpdir, "data2.pt")
            collect_trajectories(1, 10, save_path=path2, seed=42)
            
            data1 = torch.load(path1)
            data2 = torch.load(path2)
            
            # Should be identical
            assert torch.allclose(data1[0]['q'], data2[0]['q'])


class TestRobustness:
    """Test model robustness to edge cases."""
    
    def test_zero_input(self):
        """Test model handles zero input."""
        model = PIND(nv=18, nu=12, hidden_dim=16)
        model.eval()
        
        batch_size = 1
        q = torch.zeros(batch_size, 19)
        v = torch.zeros(batch_size, 18)
        u = torch.zeros(batch_size, 12)
        M_nom = torch.eye(18).unsqueeze(0)
        h_nom = torch.zeros(batch_size, 18)
        p_feet = torch.zeros(batch_size, 4, 3)
        J_feet = torch.zeros(batch_size, 4, 3, 18)
        
        with torch.no_grad():
            v_dot, lambdas, delta_tau = model.forward_acceleration(
                q, v, u, M_nom, h_nom, p_feet, J_feet
            )
        
        assert not torch.isnan(v_dot).any()
        assert not torch.isinf(v_dot).any()
    
    def test_large_input(self):
        """Test model handles large input values."""
        model = PIND(nv=18, nu=12, hidden_dim=16)
        model.eval()
        
        batch_size = 1
        q = torch.randn(batch_size, 19) * 100
        v = torch.randn(batch_size, 18) * 100
        u = torch.randn(batch_size, 12) * 100
        M_nom = torch.eye(18).unsqueeze(0) * 100
        h_nom = torch.randn(batch_size, 18) * 100
        p_feet = torch.randn(batch_size, 4, 3) * 10
        J_feet = torch.randn(batch_size, 4, 3, 18)
        
        with torch.no_grad():
            v_dot, lambdas, delta_tau = model.forward_acceleration(
                q, v, u, M_nom, h_nom, p_feet, J_feet
            )
        
        # Should still produce finite values
        assert torch.isfinite(v_dot).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
