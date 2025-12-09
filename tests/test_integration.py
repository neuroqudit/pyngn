import torch
import pytest
from pyngn.synapse import DelayBuffer
from pyngn.neuron import LIFLayer

def test_delay_consistency():
    """
    Test if spikes are retrieved with the correct delay.
    Neuron A fires at t=0. Connection A->B has delay 5.
    B should see the spike at t=5.
    """
    n_neurons = 1
    max_delay = 10
    buffer = DelayBuffer(n_neurons, max_delay)
    
    # Delay matrix [1, 1] (since we test retrieval, we can simulate 1 pre, 1 post)
    # Let's say we have 1 neuron projecting to itself or another with delay 5
    delay_matrix = torch.tensor([[5]])
    
    # t=0: Spike
    buffer.push(torch.tensor([1.0]))
    
    # Check immediate retrieval (should be 0 because delay is 5)
    # At t=1 (next step), we check what arrives.
    # Wait, the loop usually is:
    # 1. Calculate input from delayed spikes (using buffer state)
    # 2. Update neurons
    # 3. Push new spikes to buffer
    
    # So at t=0, we push spike.
    # At t=1, we query. (write_idx has advanced)
    # ...
    # At t=5, we query.
    
    # Let's simulate the steps
    # t=0: Push spike. write_idx becomes 1.
    
    # t=1: Query. indices = (1 - 5) % 10 = -4 % 10 = 6. Buffer[6] is 0.
    assert buffer.get_delayed_spikes(delay_matrix).item() == 0.0
    buffer.push(torch.tensor([0.0])) # write_idx -> 2
    
    # t=2: Query. indices = (2 - 5) % 10 = 7. Buffer[7] is 0.
    assert buffer.get_delayed_spikes(delay_matrix).item() == 0.0
    buffer.push(torch.tensor([0.0])) # write_idx -> 3
    
    # ...
    # t=5: Query. indices = (5 - 5) % 10 = 0. Buffer[0] is 1.0!
    # But wait, if we push at t=0, that's the spike generated at t=0.
    # If delay is 5, it should arrive at t=5.
    # So at step 5, we should see it.
    
    for t in range(3, 6): # t=3, 4, 5
        out = buffer.get_delayed_spikes(delay_matrix).item()
        if t == 5:
            assert out == 1.0
        else:
            assert out == 0.0
        buffer.push(torch.tensor([0.0]))

def test_buffer_overwrite():
    """Verify circular buffer overwrites correctly."""
    n_neurons = 1
    max_delay = 3
    buffer = DelayBuffer(n_neurons, max_delay)
    delay_matrix = torch.tensor([[2]])
    
    # t=0: Spike 1. Buffer[0] = 1. write_idx -> 1
    buffer.push(torch.tensor([1.0]))
    
    # t=1: Spike 0. Buffer[1] = 0. write_idx -> 2
    buffer.push(torch.tensor([0.0]))
    
    # t=2: Spike 0. Buffer[2] = 0. write_idx -> 0
    buffer.push(torch.tensor([0.0]))
    
    # t=3: Spike 1 (New). Buffer[0] should be overwritten.
    # Before push, check if we can still retrieve the old spike at t=0?
    # At t=3, delay 2 means looking at t=1. Buffer[1]=0.
    # Delay 3 means looking at t=0. Buffer[0]=1.
    
    # Overwrite
    buffer.push(torch.tensor([1.0])) # Buffer[0] = 1 (new). Old is gone.
    
    # Now at t=4.
    # Query delay 2. indices = (4 - 2) % 3 = 2. Buffer[2] = 0.
    # Query delay 3. indices = (4 - 3) % 3 = 1. Buffer[1] = 0.
    # Query delay 1. indices = (4 - 1) % 3 = 0. Buffer[0] = 1 (the new one).
    
    d_mat = torch.tensor([[1]])
    assert buffer.get_delayed_spikes(d_mat).item() == 1.0
