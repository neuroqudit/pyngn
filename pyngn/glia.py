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

class MicrogliaController(nn.Module):
    """
    Microglia Controller for Structural Plasticity (Pruning).
    
    Dynamics:
    1. Synaptic Health Tracking (Hebbian-like):
       H[t+1] = H[t] + eta_micro * (pre[t] * post[t] - decay * H[t])
       (Synapses that successfully drive post-synaptic spikes are "healthy")
       
    2. Pruning:
       If H_ij < threshold, M_ij = 0 (Prune)
       
    3. Regeneration (Optional/Simplified):
       Randomly re-enable synapses with low probability to explore topology?
       For now, we implement Pruning only.
    """
    def __init__(self, n_neurons, pruning_threshold=0.1, eta_micro=0.01, health_decay=0.1, device='cpu'):
        super().__init__()
        self.n_neurons = n_neurons
        self.pruning_threshold = pruning_threshold
        self.eta_micro = eta_micro
        self.health_decay = health_decay
        self.device = device
        
        # Synaptic Health Matrix: [n_neurons, n_neurons]
        # Initialized to 1.0 (fully healthy)
        self.health = torch.ones(n_neurons, n_neurons, device=device)
        
        # Connectivity Mask: [n_neurons, n_neurons]
        # 1 = Connected, 0 = Pruned
        self.mask = torch.ones(n_neurons, n_neurons, device=device)
        
    def update(self, pre_spikes, post_spikes):
        """
        Update synaptic health and prune weak synapses.
        
        Args:
            pre_spikes (torch.Tensor): [n_neurons] binary spike vector (t)
            post_spikes (torch.Tensor): [n_neurons] binary spike vector (t)
            
        Returns:
            torch.Tensor: Updated mask M[t+1]
        """
        if isinstance(pre_spikes, torch.Tensor):
            pre_spikes = pre_spikes.to(self.device)
        if isinstance(post_spikes, torch.Tensor):
            post_spikes = post_spikes.to(self.device)
            
        # Hebbian term: pre[i] * post[j]
        # We want matrix [i, j] where i is pre, j is post?
        # Usually weights are W[post, pre] or W[pre, post]?
        # In neuron.py: sum_{j} M_{ij} * w_{ij} * s_j
        # This implies i is post, j is pre.
        # So W is [n_post, n_pre] or we index W[i, j].
        # Let's assume W[i, j] means connection j -> i (pre=j, post=i).
        # So Hebbian = pre[j] * post[i]
        
        # Outer product: post.unsqueeze(1) * pre.unsqueeze(0) -> [post, pre]
        # Let's stick to W[i, j] = j -> i
        
        hebbian = torch.outer(post_spikes, pre_spikes)
        
        # Update Health
        # H[t+1] = H[t] + eta * (Hebbian - decay * H[t])
        delta_h = self.eta_micro * (hebbian - self.health_decay * self.health)
        
        # Only update health for currently active connections? 
        # Or do we track health even for pruned ones (latent)?
        # Let's track for all, but maybe decay dominates if disconnected.
        self.health = self.health + delta_h
        
        # Clamp Health to [0, 1]
        self.health = torch.clamp(self.health, 0.0, 1.0)
        
        # Pruning
        # If H < threshold, M = 0
        # If H >= threshold, M = 1 (Recovery? Or once pruned, stays pruned?)
        # "Structural Plasticity" usually implies both.
        # Let's allow recovery if health goes back up (e.g. random fluctuations or if we add noise).
        # But with just decay, it won't go back up if M=0 implies no transmission?
        # Wait, if M=0, pre*post might still happen due to other inputs!
        # So "Latent" synapses can recover if correlated activity appears.
        
        self.mask = (self.health >= self.pruning_threshold).float()
        
        return self.mask

    def reset_state(self):
        self.health.fill_(1.0)
        self.mask.fill_(1.0)
