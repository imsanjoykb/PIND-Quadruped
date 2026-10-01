import torch
import pytest
from pind.models.pind import PIND, ActuatorResidualMLP, ContactComplianceNetwork
from pind.models.baselines import BlackBoxMLP, NeuralODEVectorField, UnconstrainedResidualMLP


class TestPIND:
    
    def setup_method(self):
        """Setup test fixtures."""
        self.batch_size = 4
        self.nv = 18
        self.nu = 12
        self.model = PIND(nv=self.nv, nu=self.nu, hidden_dim=64)
    
    def test_model_creation(self):
        """Test model can be created."""
        assert self.model is not None
        assert self.model.nv == self.nv
        assert self.model.nu == self.nu
    
    def test_forward_pass(self):
        """Test forward pass with random data."""
        q = torch.randn(self.batch_size, 19)  # Position (with quaternion)
        v = torch.randn(self.batch_size, self.nv)
        u = torch.randn(self.batch_size, self.nu)
        M_nom = torch.eye(self.nv).unsqueeze(0).repeat(self.batch_size, 1, 1)
        h_nom = torch.zeros(self.batch_size, self.nv)
        p_feet = torch.randn(self.batch_size, 4, 3)
        J_feet = torch.randn(self.batch_size, 4, 3, self.nv)
        
        v_dot, lambdas, delta_tau = self.model.forward_acceleration(
            q, v, u, M_nom, h_nom, p_feet, J_feet
        )
        
        assert v_dot.shape == (self.batch_size, self.nv)
        assert len(lambdas) == 4
        assert delta_tau.shape == (self.batch_size, self.nu)
    
    def test_loss_computation(self):
        """Test loss computation."""
        q = torch.randn(self.batch_size, 19)
        v = torch.randn(self.batch_size, self.nv)
        u = torch.randn(self.batch_size, self.nu)
        M_nom = torch.eye(self.nv).unsqueeze(0).repeat(self.batch_size, 1, 1)
        h_nom = torch.zeros(self.batch_size, self.nv)
        p_feet = torch.randn(self.batch_size, 4, 3)
        J_feet = torch.randn(self.batch_size, 4, 3, self.nv)
        v_dot_target = torch.randn(self.batch_size, self.nv)
        
        v_dot_pred, lambdas, delta_tau = self.model.forward_acceleration(
            q, v, u, M_nom, h_nom, p_feet, J_feet
        )
        
        loss, metrics = self.model.compute_composite_loss(
            v_dot_pred, v_dot_target, lambdas, p_feet,
            M_nom, h_nom, u, delta_tau, J_feet
        )
        
        assert loss.item() >= 0
        assert 'l_data' in metrics
        assert 'l_dyn' in metrics


class TestBaselines:
    
    def test_black_box_mlp(self):
        """Test BlackBoxMLP."""
        model = BlackBoxMLP(state_dim=37, act_dim=12, hidden_dim=128)
        x = torch.randn(4, 37)
        u = torch.randn(4, 12)
        
        output = model(x, u)
        assert output.shape == (4, 37)
    
    def test_neural_ode(self):
        """Test NeuralODEVectorField."""
        model = NeuralODEVectorField(state_dim=37, act_dim=12)
        x = torch.randn(4, 37)
        u = torch.randn(4, 12)
        
        output = model(x, u)
        assert output.shape == (4, 37)
    
    def test_residual_mlp(self):
        """Test UnconstrainedResidualMLP."""
        model = UnconstrainedResidualMLP(state_dim=37, act_dim=12, nv=18)
        x = torch.randn(4, 37)
        u = torch.randn(4, 12)
        v_dot_nom = torch.randn(4, 18)
        
        output = model(x, u, v_dot_nom)
        assert output.shape == (4, 18)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
