import torch
import torch.nn as nn
from .neuron import LIFLayer
from .synapse import DelayBuffer
from .glia import AstrocyteController, MicrogliaController
from .readout import ReadoutLayer

class TriglialReservoir(nn.Module):
    """
    3-Glia Spiking Neural Network (3GSNN) Reservoir.
    
    Integrates:
    - Neurons: LIF Dynamics
    - Synapses: Delays (Oligodendrocytes) + Weights
    - Astrocytes: Homeostatic Gain Control
    - Microglia: Structural Plasticity (Pruning)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 dt=1.0, max_delay=10, 
                 astro_params=None, micro_params=None,
                 device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dt = dt
        self.device = device
        
        # 1. Neuron Layer
        self.neurons = LIFLayer(hidden_dim, dt=dt, device=device)
        
        # 2. Input Weights (Fixed random)
        # Input -> Hidden
        self.input_weights = torch.randn(input_dim, hidden_dim, device=device) * 0.1
        
        # 3. Recurrent Weights (Reservoir)
        # Hidden -> Hidden
        # Initialize with sparse connectivity?
        self.recurrent_weights = torch.randn(hidden_dim, hidden_dim, device=device) * 0.1
        
        # 4. Delays (Oligodendrocytes)
        self.delay_buffer = DelayBuffer(hidden_dim, max_delay=max_delay, device=device)
        # Random delays for now
        self.delays = torch.randint(1, max_delay, (hidden_dim, hidden_dim), device=device)
        
        # 5. Astrocytes (Homeostasis)
        if astro_params is None:
            astro_params = {}
        self.astrocyte = AstrocyteController(hidden_dim, device=device, **astro_params)
        
        # 6. Microglia (Pruning)
        if micro_params is None:
            micro_params = {}
        self.microglia = MicrogliaController(hidden_dim, device=device, **micro_params)
        
        # 7. Readout
        self.readout = ReadoutLayer(hidden_dim, output_dim, device=device)
        
    def forward(self, x, steps=100, training_readout=False, target=None, return_state=False):
        """
        Run reservoir simulation.
        
        Args:
            x (torch.Tensor): Input spike train [steps, input_dim]
            steps (int): Simulation steps (if x is not time-series)
            training_readout (bool): If True, update readout weights online (not recommended for Ridge)
            target (torch.Tensor): Target output for training
            return_state (bool): If True, return (prediction, final_state) tuple
            
        Returns:
            torch.Tensor or tuple: Readout output, or (output, state)
        """
        # Assuming x is [steps, input_dim]
        if x.dim() == 2:
            seq_len, _ = x.shape
        else:
            raise ValueError("Input must be [steps, input_dim]")
            
        self.reset_state()
        
        reservoir_states = []
        
        for t in range(seq_len):
            input_t = x[t] # [input_dim]
            
            # 1. Input Current
            # I_in = x(t) @ W_in
            i_input = torch.matmul(input_t, self.input_weights) # [hidden_dim]
            
            # 2. Recurrent Current (Delayed)
            # Get delayed spikes
            delayed_spikes = self.delay_buffer.get_delayed_spikes(self.delays) # [hidden, hidden]
            
            # Apply Microglia Mask
            # W_eff = W_rec * Mask
            w_effective = self.recurrent_weights * self.microglia.mask
            
            # I_rec = sum(W_eff * delayed_spikes)
            i_rec = (w_effective * delayed_spikes).sum(dim=0) # [hidden_dim]
            
            # Total Synaptic Current
            i_syn = i_input + i_rec
            
            # 3. Astrocyte Modulation
            # I_total = I_syn * Gamma + I_ext
            i_total = i_syn * self.astrocyte.gamma
            
            # 4. Neuron Update
            spikes = self.neurons.forward(i_syn=i_total)
            
            # 5. Update Glia & Delays
            # Astrocyte
            self.astrocyte.update(spikes)
            
            # Microglia (Hebbian-like)
            self.microglia.update(spikes, spikes) 
            
            # 6. Push to Delay Buffer
            self.delay_buffer.push(spikes)
            
            # Collect state (Voltage)
            reservoir_states.append(self.neurons.v.clone())
            
        # Stack states: [steps, hidden_dim]
        states_tensor = torch.stack(reservoir_states)
        
        # Readout: Mean pooling
        final_state = states_tensor.mean(dim=0).unsqueeze(0) # [1, hidden_dim]
        
        if training_readout and target is not None:
            # Fit readout (Online/Incremental if supported, but here it's batch Ridge)
            # Warning: This overwrites weights based on single sample!
            self.readout.fit(final_state, target.unsqueeze(0))
            
        prediction = self.readout(final_state)
        
        if return_state:
            return prediction, final_state
            
        return prediction

    def reset_state(self):
        self.neurons.reset_state()
        self.delay_buffer.reset_state()
        self.astrocyte.reset_state()
        self.microglia.reset_state()
