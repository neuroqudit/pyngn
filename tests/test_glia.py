import torch
import pytest
from pyngn.glia import AstrocyteController

def test_gain_up():
    """Test if gain increases when activity is zero (below target)."""
    n_neurons = 1
    target = 0.5
    astro = AstrocyteController(n_neurons, target_rate=target, eta_astro=0.1, beta_ca=0.0)
    
    # Initial gain is 1.0
    # Input spikes = 0
    # Calcium = 0 * 0 + 0 = 0
    # Delta = 0.1 * (0.5 - 0) = 0.05
    # New Gamma = 1.0 + 0.05 = 1.05
    
    gamma = astro.update(torch.tensor([0.0]))
    assert torch.isclose(gamma, torch.tensor([1.05]))
    
    # Next step, still 0 input
    # Calcium = 0
    # Delta = 0.05
    # New Gamma = 1.10
    gamma = astro.update(torch.tensor([0.0]))
    assert torch.isclose(gamma, torch.tensor([1.10]))

def test_gain_down():
    """Test if gain decreases when activity is high (above target)."""
    n_neurons = 1
    target = 0.5
    astro = AstrocyteController(n_neurons, target_rate=target, eta_astro=0.1, beta_ca=0.0)
    
    # Input spikes = 1 (High activity)
    # Calcium = 0 * 0 + 1 = 1
    # Delta = 0.1 * (0.5 - 1.0) = -0.05
    # New Gamma = 1.0 - 0.05 = 0.95
    
    gamma = astro.update(torch.tensor([1.0]))
    assert torch.isclose(gamma, torch.tensor([0.95]))

def test_clamping():
    """Test if gain stays within bounds."""
    n_neurons = 1
    astro = AstrocyteController(n_neurons, gamma_min=0.5, gamma_max=1.5, eta_astro=1.0, target_rate=2.0, beta_ca=0.0)
    
    # Force increase
    # Input 0. Ca=0. Delta = 1.0 * (2.0 - 0) = 2.0
    # Gamma = 1.0 + 2.0 = 3.0 -> Clamped to 1.5
    
    gamma = astro.update(torch.tensor([0.0]))
    assert gamma.item() == 1.5
    
    # Force decrease
    astro = AstrocyteController(n_neurons, gamma_min=0.5, gamma_max=1.5, eta_astro=1.0, target_rate=-1.0, beta_ca=0.0)
    # Input 0. Ca=0. Delta = 1.0 * (-1.0 - 0) = -1.0
    # Gamma = 1.0 - 1.0 = 0.0 -> Clamped to 0.5
    
    gamma = astro.update(torch.tensor([0.0]))
    assert gamma.item() == 0.5
