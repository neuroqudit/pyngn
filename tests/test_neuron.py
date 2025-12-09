import torch
import pytest
from pyngn.neuron import LIFLayer

def test_decay():
    """Test if voltage decays exponentially without input."""
    dt = 1.0
    tau = 10.0
    layer = LIFLayer(n_neurons=1, tau_m=tau, dt=dt, v_th=10.0)
    
    # Set initial voltage
    layer.v = torch.tensor([5.0])
    layer.s = torch.tensor([0.0]) # No spike
    
    # Step without input
    layer.forward(i_syn=0.0)
    
    expected_v = 5.0 * torch.exp(torch.tensor(-dt/tau))
    assert torch.isclose(layer.v, expected_v, atol=1e-4)

def test_fire_and_reset():
    """Test if neuron fires when threshold is crossed and resets afterwards."""
    layer = LIFLayer(n_neurons=1, v_th=1.0, tau_m=10.0)
    
    # Inject strong current to force spike
    # If v starts at 0, and we add 1.5, it should spike
    s = layer.forward(i_syn=1.5)
    
    assert s.item() == 1.0
    assert layer.v.item() == 1.5 # Voltage is high at the moment of spike
    
    # Next step: should reset
    # v[t+1] = alpha * v[t] * (1 - s[t]) + input
    # If input is 0, v should be 0 because (1 - 1) = 0
    layer.forward(i_syn=0.0)
    assert layer.v.item() == 0.0
    assert layer.s.item() == 0.0

def test_constant_current_spiking():
    """Test regular spiking under constant current."""
    layer = LIFLayer(n_neurons=1, v_th=1.0, tau_m=10.0)
    
    spikes = []
    # Inject current just above threshold? Or accumulate?
    # With alpha < 1, v_inf = I * R (here R effectively related to 1/(1-alpha) approx)
    # Actually if we just add I every step: v[t+1] = alpha*v[t] + I
    # Fixed point: v = alpha*v + I => v(1-alpha) = I => v = I / (1-alpha)
    
    # alpha = exp(-0.1) approx 0.9048
    # 1 - alpha approx 0.095
    # If I = 0.2, v_inf = 0.2 / 0.095 > 2.0 > v_th
    # So it should spike eventually
    
    for _ in range(50):
        s = layer.forward(i_syn=0.2)
        spikes.append(s.item())
        
    assert sum(spikes) > 0 # Should have spiked at least once
    
    # Check if it spikes periodically
    # It takes some steps to reach threshold, then resets, then climbs again
    # So we expect a pattern like 0,0,0,1,0,0,0,1...
    
    # Verify it's not always 1 (unless current is huge)
    assert sum(spikes) < 50

def test_refractory_logic():
    """Verify reset logic effectively acts as refractory period of 1 step."""
    layer = LIFLayer(n_neurons=1, v_th=1.0)
    
    # Force spike
    layer.forward(i_syn=2.0)
    assert layer.s.item() == 1.0
    
    # Next step, even with input, the previous voltage component is killed
    # v[t+1] = alpha * v[t] * 0 + I
    # So v[t+1] = I
    # If I < v_th, it won't spike immediately again if it relies on accumulation
    
    # Case: I is small, relies on integration
    # If we didn't reset, v would be 2.0 * alpha + 0.5 = 2.3 -> spike again
    # With reset, v = 0 + 0.5 = 0.5 -> no spike
    
    s_next = layer.forward(i_syn=0.5)
    assert s_next.item() == 0.0
    assert layer.v.item() == 0.5
