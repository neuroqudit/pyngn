import torch
import torch.nn as nn

class AstrocyteController(nn.Module):
    """
    Astrocyte Controller for Homeostatic Gain Regulation.
    
    Dynamics:
    1. Calcium Integration:
       Ca[t+1] = beta_Ca * Ca[t] + s[t]
       (Low-pass filter of neural activity)
       
    2. Gain Control (PID-like):
       gamma[t+1] = gamma[t] + eta_astro * (target_rate - Ca[t])
       Restricted to [gamma_min, gamma_max]
       
    The gain gamma modulates the recurrent weights in the reservoir.
    """
    def __init__(self, n_neurons, target_rate=0.1, eta_astro=0.001, beta_ca=0.9, 
                 gamma_min=0.5, gamma_max=2.0, device='cpu'):
        super().__init__()
        self.n_neurons = n_neurons
        self.target_rate = target_rate
        self.eta_astro = eta_astro
        self.beta_ca = beta_ca
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.device = device
        
        # State tensors
        self.calcium = torch.zeros(n_neurons, device=device)
        self.gamma = torch.ones(n_neurons, device=device) # Start with gain 1.0
        
    def update(self, spikes):
        """
        Update astrocyte state based on neural activity.
        
        Args:
            spikes (torch.Tensor): [n_neurons] binary spike vector s[t]
            
        Returns:
            torch.Tensor: Updated gain gamma[t+1]
        """
        if isinstance(spikes, torch.Tensor):
            spikes = spikes.to(self.device)
            
        # 1. Integrate Calcium
        # Ca[t+1] = beta * Ca[t] + s[t]
        # Note: s[t] is usually 0 or 1. If beta is close to 1, Ca accumulates.
        # If we want Ca to represent a rate, we might want (1-beta)*s[t]?
        # The user specified: Ca[t+1] = beta_Ca * Ca[t] + s[t]
        # This acts as a leaky integrator.
        self.calcium = self.beta_ca * self.calcium + spikes
        
        # 2. Update Gain
        # gamma[t+1] = gamma[t] + eta * (target - Ca[t])
        # If Ca < target, term is positive -> gain increases (to boost activity)
        # If Ca > target, term is negative -> gain decreases (to suppress activity)
        delta_gamma = self.eta_astro * (self.target_rate - self.calcium)
        self.gamma = self.gamma + delta_gamma
        
        # Clamp gain
        self.gamma = torch.clamp(self.gamma, self.gamma_min, self.gamma_max)
        
        return self.gamma

    def reset_state(self):
        self.calcium.zero_()
        self.gamma.fill_(1.0)
