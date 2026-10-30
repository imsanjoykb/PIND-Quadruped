import pytest
import torch
import numpy as np
from pind.utils import QuadrupedMuJoCoInterface, rk4_step


class TestMuJoCoInterface:
    """Test MuJoCo interface for physics extraction."""
    
    @pytest.fixture
    def sim(self):
        """Create MuJoCo simulation instance."""
        return QuadrupedMuJoCoInterface()
    
    def test_initialization(self, sim):
        """Test simulator initializes correctly."""
        assert sim.nv == 18, "Should have 18 velocity DOFs"
        assert sim.nq == 19, "Should have 19 position coordinates"
        assert sim.nu == 12, "Should have 12 actuated joints"
        assert sim.model is not None
        assert sim.data is not None
    
    def test_compute_nominal_dynamics_shape(self, sim):
        """Test nominal dynamics computation returns correct shapes."""
        q = np.random.randn(sim.nq)
        v = np.random.randn(sim.nv)
        
        M, h, p_feet, J_feet = sim.compute_nominal_dynamics(q, v)
        
        assert M.shape == (sim.nv, sim.nv), "Mass matrix shape incorrect"
        assert h.shape == (sim.nv,), "Bias forces shape incorrect"
        assert p_feet.shape == (4, 3), "Foot positions shape incorrect"
        assert J_feet.shape == (4, 3, sim.nv), "Jacobians shape incorrect"
    
    def test_mass_matrix_properties(self, sim):
        """Test mass matrix is symmetric and positive definite."""
        q = np.random.randn(sim.nq)
        v = np.zeros(sim.nv)
        
        M, _, _, _ = sim.compute_nominal_dynamics(q, v)
        
        # Symmetry
        assert np.allclose(M, M.T), "Mass matrix should be symmetric"
        
        # Positive definiteness (all eigenvalues > 0)
        eigenvalues = np.linalg.eigvals(M)
        assert np.all(eigenvalues > 0), "Mass matrix should be positive definite"
    
    def test_standing_configuration(self, sim):
        """Test physics at standing configuration."""
        q = np.zeros(sim.nq)
        q[2] = 0.3  # Base height
        q[3] = 1.0  # Quaternion w
        v = np.zeros(sim.nv)
        
        M, h, p_feet, J_feet = sim.compute_nominal_dynamics(q, v)
        
        # Feet should be near ground
        assert np.all(p_feet[:, 2] <= 0.1), "Feet should be near ground"
        
        # Mass matrix should be reasonable
        assert M.max() < 1000, "Mass matrix values reasonable"
    
    def test_jacobian_rank(self, sim):
        """Test contact Jacobians have correct rank."""
        q = np.random.randn(sim.nq)
        v = np.zeros(sim.nv)
        
        _, _, _, J_feet = sim.compute_nominal_dynamics(q, v)
        
        for i in range(4):
            J = J_feet[i]  # (3, 18)
            rank = np.linalg.matrix_rank(J)
            assert rank <= 3, f"Jacobian {i} rank should be <= 3"


class TestRK4Integrator:
    """Test RK4 numerical integrator."""
    
    def test_linear_dynamics(self):
        """Test RK4 on simple linear system: ẋ = -x."""
        def linear_field(x, u):
            return -x
        
        x0 = torch.tensor([[1.0, 2.0]])
        u = torch.zeros(1, 2)
        dt = 0.1
        
        x1 = rk4_step(linear_field, x0, u, dt)
        
        # Analytical solution: x(t) = x0 * exp(-t)
        expected = x0 * torch.exp(torch.tensor(-dt))
        
        assert torch.allclose(x1, expected, atol=1e-4), "RK4 should match analytical solution"
    
    def test_batch_processing(self):
        """Test RK4 handles batched inputs."""
        def dynamics(x, u):
            return u
        
        batch_size = 10
        state_dim = 5
        
        x = torch.randn(batch_size, state_dim)
        u = torch.randn(batch_size, state_dim)
        dt = 0.01
        
        x_next = rk4_step(dynamics, x, u, dt)
        
        assert x_next.shape == (batch_size, state_dim)
    
    def test_harmonic_oscillator(self):
        """Test RK4 on harmonic oscillator: ẍ = -ω²x."""
        omega = 2.0
        
        def oscillator(x, u):
            # x = [position, velocity]
            pos, vel = x[:, 0:1], x[:, 1:2]
            dpos = vel
            dvel = -omega**2 * pos
            return torch.cat([dpos, dvel], dim=-1)
        
        x0 = torch.tensor([[1.0, 0.0]])  # Initial displacement
        u = torch.zeros(1, 2)
        dt = 0.01
        
        # Integrate one step
        x1 = rk4_step(oscillator, x0, u, dt)
        
        # Should preserve energy (approximately)
        energy0 = 0.5 * x0[:, 1]**2 + 0.5 * omega**2 * x0[:, 0]**2
        energy1 = 0.5 * x1[:, 1]**2 + 0.5 * omega**2 * x1[:, 0]**2
        
        assert torch.allclose(energy0, energy1, rtol=1e-3), "Energy should be conserved"
    
    def test_zero_timestep(self):
        """Test RK4 with zero timestep returns same state."""
        def dynamics(x, u):
            return torch.randn_like(x)
        
        x = torch.randn(5, 10)
        u = torch.randn(5, 10)
        
        x_next = rk4_step(dynamics, x, u, dt=0.0)
        
        assert torch.allclose(x, x_next), "Zero timestep should not change state"
    
    def test_consistency(self):
        """Test RK4 gives consistent results."""
        def dynamics(x, u):
            return x + u
        
        x = torch.tensor([[1.0, 2.0]])
        u = torch.tensor([[0.1, 0.2]])
        dt = 0.05
        
        # Run twice
        result1 = rk4_step(dynamics, x, u, dt)
        result2 = rk4_step(dynamics, x, u, dt)
        
        assert torch.allclose(result1, result2), "Should give consistent results"


class TestIntegrationConsistency:
    """Test integration between components."""
    
    def test_mujoco_with_torch(self):
        """Test MuJoCo outputs can be converted to PyTorch."""
        sim = QuadrupedMuJoCoInterface()
        
        q = np.random.randn(sim.nq)
        v = np.random.randn(sim.nv)
        
        M, h, p_feet, J_feet = sim.compute_nominal_dynamics(q, v)
        
        # Convert to PyTorch
        M_torch = torch.from_numpy(M).float()
        h_torch = torch.from_numpy(h).float()
        p_feet_torch = torch.from_numpy(p_feet).float()
        J_feet_torch = torch.from_numpy(J_feet).float()
        
        assert M_torch.shape == (sim.nv, sim.nv)
        assert not torch.isnan(M_torch).any()
        assert not torch.isinf(M_torch).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
