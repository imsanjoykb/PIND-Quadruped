"""
Test suite for data generation and loading.

Tests trajectory generation and dataset functionality.
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path
from pind.data import QuadrupedDataset, collect_trajectories


class TestDataGeneration:
    """Test data generation functionality."""
    
    def test_collect_trajectories_basic(self):
        """Test basic trajectory collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_dataset.pt")
            
            # Generate small dataset
            collect_trajectories(
                num_trajectories=2,
                trajectory_length=10,
                dt=0.02,
                save_path=output_path,
                seed=42
            )
            
            # Check file exists
            assert os.path.exists(output_path), "Dataset file should be created"
            
            # Load and check
            dataset = torch.load(output_path)
            assert len(dataset) == 2 * 10, "Should have 20 samples"
    
    def test_collect_trajectories_reproducibility(self):
        """Test trajectory generation is reproducible with seed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = os.path.join(tmpdir, "data1.pt")
            path2 = os.path.join(tmpdir, "data2.pt")
            
            # Generate twice with same seed
            collect_trajectories(2, 5, save_path=path1, seed=42)
            collect_trajectories(2, 5, save_path=path2, seed=42)
            
            data1 = torch.load(path1)
            data2 = torch.load(path2)
            
            # Compare first samples
            for key in data1[0].keys():
                if isinstance(data1[0][key], torch.Tensor):
                    assert torch.allclose(data1[0][key], data2[0][key]), \
                        f"Key {key} should match with same seed"


class TestQuadrupedDataset:
    """Test QuadrupedDataset class."""
    
    @pytest.fixture
    def dataset_path(self):
        """Create temporary dataset."""
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "test_dataset.pt")
        
        collect_trajectories(
            num_trajectories=2,
            trajectory_length=10,
            save_path=path,
            seed=42
        )
        
        yield path
        
        # Cleanup
        import shutil
        shutil.rmtree(tmpdir)
    
    def test_dataset_initialization(self, dataset_path):
        """Test dataset loads correctly."""
        dataset = QuadrupedDataset(dataset_path)
        assert len(dataset) > 0
    
    def test_dataset_getitem(self, dataset_path):
        """Test getting items from dataset."""
        dataset = QuadrupedDataset(dataset_path)
        
        sample = dataset[0]
        
        # Check all required keys present
        required_keys = ['q', 'v', 'u', 'q_next', 'v_next', 'v_dot_gt',
                        'M_nom', 'h_nom', 'p_feet', 'J_feet']
        
        for key in required_keys:
            assert key in sample, f"Missing key: {key}"
            assert isinstance(sample[key], torch.Tensor), f"{key} should be tensor"
    
    def test_dataset_shapes(self, dataset_path):
        """Test dataset tensors have correct shapes."""
        dataset = QuadrupedDataset(dataset_path)
        sample = dataset[0]
        
        assert sample['q'].shape == (19,), "Position shape incorrect"
        assert sample['v'].shape == (18,), "Velocity shape incorrect"
        assert sample['u'].shape == (12,), "Control shape incorrect"
        assert sample['v_dot_gt'].shape == (18,), "Acceleration shape incorrect"
        assert sample['M_nom'].shape == (18, 18), "Mass matrix shape incorrect"
        assert sample['h_nom'].shape == (18,), "Bias forces shape incorrect"
        assert sample['p_feet'].shape == (4, 3), "Foot positions shape incorrect"
        assert sample['J_feet'].shape == (4, 3, 18), "Jacobians shape incorrect"
    
    def test_dataset_iteration(self, dataset_path):
        """Test iterating through dataset."""
        dataset = QuadrupedDataset(dataset_path)
        
        count = 0
        for sample in dataset:
            count += 1
            assert 'q' in sample
            assert 'v' in sample
        
        assert count == len(dataset)
    
    def test_dataset_values_finite(self, dataset_path):
        """Test dataset contains finite values."""
        dataset = QuadrupedDataset(dataset_path)
        sample = dataset[0]
        
        for key, value in sample.items():
            assert torch.isfinite(value).all(), f"{key} contains non-finite values"
    
    def test_mass_matrix_properties(self, dataset_path):
        """Test mass matrix in dataset is valid."""
        dataset = QuadrupedDataset(dataset_path)
        sample = dataset[0]
        
        M = sample['M_nom']
        
        # Should be symmetric
        assert torch.allclose(M, M.T, atol=1e-5), "Mass matrix should be symmetric"
        
        # Should be positive definite
        eigenvalues = torch.linalg.eigvals(M).real
        assert (eigenvalues > 0).all(), "Mass matrix should be positive definite"


class TestDataLoading:
    """Test data loading with PyTorch DataLoader."""
    
    @pytest.fixture
    def dataset_path(self):
        """Create temporary dataset."""
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "test_dataset.pt")
        
        collect_trajectories(2, 10, save_path=path, seed=42)
        yield path
        
        import shutil
        shutil.rmtree(tmpdir)
    
    def test_dataloader_batching(self, dataset_path):
        """Test batching with DataLoader."""
        from torch.utils.data import DataLoader
        
        def collate_fn(batch):
            return {k: torch.stack([b[k] for b in batch]) for k in batch[0].keys()}
        
        dataset = QuadrupedDataset(dataset_path)
        loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
        
        batch = next(iter(loader))
        
        assert batch['q'].shape[0] == 4, "Batch size should be 4"
        assert batch['q'].shape[1] == 19, "Position dim should be 19"
        assert batch['M_nom'].shape == (4, 18, 18), "Batched mass matrix shape"
    
    def test_dataloader_shuffling(self, dataset_path):
        """Test shuffling works."""
        from torch.utils.data import DataLoader
        
        dataset = QuadrupedDataset(dataset_path)
        
        loader1 = DataLoader(dataset, batch_size=1, shuffle=True)
        loader2 = DataLoader(dataset, batch_size=1, shuffle=True)
        
        # Get first samples
        sample1 = next(iter(loader1))
        sample2 = next(iter(loader2))
        
        # With shuffle, they might be different (not guaranteed but likely)
        # Just check they're valid
        assert 'q' in sample1
        assert 'q' in sample2


class TestDataConsistency:
    """Test data consistency and physics validity."""
    
    @pytest.fixture
    def dataset_path(self):
        """Create dataset."""
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "test_dataset.pt")
        collect_trajectories(1, 20, save_path=path, seed=42)
        yield path
        import shutil
        shutil.rmtree(tmpdir)
    
    def test_temporal_consistency(self, dataset_path):
        """Test q_next matches next sample's q."""
        dataset = QuadrupedDataset(dataset_path)
        
        if len(dataset) > 1:
            sample1 = dataset[0]
            sample2 = dataset[1]
            
            # q_next of sample 0 should match q of sample 1
            assert torch.allclose(sample1['q_next'], sample2['q'], atol=1e-5)
    
    def test_acceleration_consistency(self, dataset_path):
        """Test acceleration is consistent with velocity change."""
        dataset = QuadrupedDataset(dataset_path)
        sample = dataset[0]
        
        dt = 0.02
        v_dot_numeric = (sample['v_next'] - sample['v']) / dt
        
        # Should be close to v_dot_gt
        assert torch.allclose(v_dot_numeric, sample['v_dot_gt'], rtol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
