import torch
import torch.nn as nn

class DelayBuffer(nn.Module):
    """
    Ring Buffer for managing heterogeneous synaptic delays (Oligodendrocytes).
    
    Dynamics:
    Stores spikes in a circular buffer of size [max_delay, n_neurons].
    Retrieves spikes based on a delay matrix D[pre, post].
    """
    def __init__(self, n_neurons, max_delay=10, device='cpu'):
        super().__init__()
        self.n_neurons = n_neurons
        self.max_delay = max_delay
        self.device = device
        
        # Buffer: [max_delay, n_neurons]
        # Rows represent time steps relative to current time t
        # Row t % max_delay is the "current" write head
        self.buffer = torch.zeros(max_delay, n_neurons, device=device)
        
        self.write_idx = 0
        
    def push(self, spikes):
        """
        Push new spikes into the buffer.
        
        Args:
            spikes (torch.Tensor): [n_neurons] binary spike vector
        """
        if isinstance(spikes, torch.Tensor):
            spikes = spikes.to(self.device)
            
        self.buffer[self.write_idx] = spikes
        
        # Advance write index
        self.write_idx = (self.write_idx + 1) % self.max_delay
        
    def get_delayed_spikes(self, delay_matrix):
        """
        Retrieve delayed spikes for each connection.
        
        Args:
            delay_matrix (torch.Tensor): [n_pre, n_post] matrix of integer delays.
                                         Values must be in [1, max_delay].
                                         
        Returns:
            torch.Tensor: [n_pre, n_post] binary matrix where entry (i, j) is 1 
                          if neuron i spiked at time t - delay_matrix[i, j].
        """
        # We need to find the index in the buffer for each connection
        # Current time is effectively self.write_idx (where we WOULD write next, 
        # so the latest written was write_idx - 1)
        
        # If delay is d, we want the spike written at t - d
        # Index = (write_idx - d) % max_delay
        
        # delay_matrix: [n_pre, n_post]
        # We want to construct a result matrix of same shape
        
        # Expand buffer to allow gathering?
        # Buffer is [max_delay, n_pre] (assuming n_neurons is n_pre)
        
        # Calculate indices for gather
        # indices[i, j] = (self.write_idx - delay_matrix[i, j]) % self.max_delay
        
        if isinstance(delay_matrix, torch.Tensor):
            delay_matrix = delay_matrix.to(self.device)
            
        # Ensure delays are valid
        # Note: delay 0 is not supported by this logic if we just wrote to write_idx-1
        # If delay=1, we want write_idx-1.
        # If delay=max_delay, we want write_idx-max_delay = write_idx
        
        indices = (self.write_idx - delay_matrix) % self.max_delay
        indices = indices.long()
        
        # Now we need to gather from buffer.
        # Buffer: [T, N_pre]
        # We want Result[i, j] = Buffer[indices[i, j], i]
        # This is a bit tricky with standard gather because we are indexing the time dimension 
        # but the index depends on both i (pre) and j (post).
        
        # Let's use advanced indexing
        # We need to select specific time indices for each neuron i.
        # Actually, for a fixed i (pre-synaptic neuron), we might have different delays to different j (post).
        # So we are looking up Buffer[time_idx, neuron_idx]
        
        # Create a grid of neuron indices
        n_pre = self.n_neurons
        n_post = delay_matrix.shape[1]
        
        # neuron_indices[i, j] = i
        neuron_indices = torch.arange(n_pre, device=self.device).unsqueeze(1).expand(n_pre, n_post)
        
        # Gather
        delayed_spikes = self.buffer[indices, neuron_indices]
        
        return delayed_spikes

    def reset_state(self):
        self.buffer.zero_()
        self.write_idx = 0
