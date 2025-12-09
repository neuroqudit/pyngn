import torch
import torch.nn as nn

class LIFLayer(nn.Module):
    """
    Layer of Leaky Integrate-and-Fire (LIF) neurons.
    
    Dynamics:
    u[t+1] = alpha * u[t] * (1 - s[t]) + I_syn[t] + I_ext[t]
    s[t] = Heaviside(u[t] - v_th)
    """
    def __init__(self, n_neurons, tau_m=20.0, v_th=1.0, dt=1.0, device='cpu'):
        super().__init__()
        self.n_neurons = n_neurons
        self.tau_m = tau_m
        self.v_th = v_th
        self.dt = dt
        self.device = device
        
        # Decay factor alpha = exp(-dt/tau_m)
        self.alpha = torch.exp(torch.tensor(-dt / tau_m))
        
        # State tensors
        self.v = torch.zeros(n_neurons, device=device)
        self.s = torch.zeros(n_neurons, device=device)
        
        # Refractory period (simplified: hard reset implies 1 step refractory if not handled otherwise)
        # For this implementation, we stick to the provided equation: reset happens at t+1 based on s[t]
        
    def forward(self, i_syn, i_ext=0.0):
        """
        Update neuron state for one time step.
        
        Args:
            i_syn (torch.Tensor): Synaptic current input [batch_size, n_neurons] or [n_neurons]
            i_ext (torch.Tensor or float): External current input
            
        Returns:
            torch.Tensor: Spikes s[t] (0 or 1)
        """
        # Ensure inputs are on the correct device
        if isinstance(i_syn, torch.Tensor):
            i_syn = i_syn.to(self.device)
        if isinstance(i_ext, torch.Tensor):
            i_ext = i_ext.to(self.device)
            
        # Update membrane potential
        # u[t+1] = alpha * u[t] * (1 - s[t]) + I_syn[t] + I_ext[t]
        # Note: self.v holds u[t] at the beginning of the call
        # self.s holds s[t-1] (spikes from previous step) which caused reset?
        # Actually, standard discrete update:
        # 1. Calculate v_new based on v_old (decayed) + input
        # 2. Check for spikes
        # 3. Reset if spiked (often done in next step or immediately)
        
        # Based on user equation: u[t+1] = alpha * u[t] * (1 - s[t]) + ...
        # This implies s[t] is the spike generated at time t.
        # So we use the PREVIOUS spike to reset the CURRENT voltage before integration?
        # Or does s[t] mean "spike that just happened"?
        # Usually:
        # v[t] = v[t-1] * decay * (1 - s[t-1]) + input
        # s[t] = (v[t] > v_th)
        
        # Let's follow the equation strictly:
        # u_new = alpha * u_old * (1 - s_old) + input
        
        self.v = self.alpha * self.v * (1 - self.s) + i_syn + i_ext
        
        # Generate spikes
        # s[t] = Theta(u[t] - v_th)
        self.s = (self.v >= self.v_th).float()
        
        # Note: The reset (1 - s[t]) will be applied in the NEXT time step's update
        # This is consistent with "Hard Reset after firing"
        
        return self.s

    def reset_state(self):
        """Reset voltage and spikes to zero."""
        self.v = torch.zeros(self.n_neurons, device=self.device)
        self.s = torch.zeros(self.n_neurons, device=self.device)
