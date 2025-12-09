import torch
import torch.nn as nn

class ReadoutLayer(nn.Module):
    """
    Readout Layer for Liquid State Machine.
    
    Implements Ridge Regression (Linear Regression with L2 regularization).
    Y = W_out * X + b
    
    Training is done in batch mode using the closed-form solution:
    W_out = (X^T * X + alpha * I)^-1 * X^T * Y_target
    """
    def __init__(self, input_dim, output_dim, alpha=1.0, device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.device = device
        
        # Weights: [input_dim, output_dim]
        # We use Linear layer for easy forward pass, but we will manually set weights
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        self.linear.to(device)
        
    def forward(self, x):
        """
        Predict output.
        
        Args:
            x (torch.Tensor): [batch_size, input_dim] reservoir states
            
        Returns:
            torch.Tensor: [batch_size, output_dim] predictions
        """
        return self.linear(x)
        
    def fit(self, states, targets):
        """
        Train readout weights using Ridge Regression.
        
        Args:
            states (torch.Tensor): [n_samples, input_dim] reservoir states (X)
            targets (torch.Tensor): [n_samples, output_dim] target outputs (Y)
        """
        if isinstance(states, torch.Tensor):
            states = states.to(self.device)
        if isinstance(targets, torch.Tensor):
            targets = targets.to(self.device)
            
        n_samples = states.shape[0]
        
        # Add bias term to states for closed-form solution
        # X_aug = [X, 1]
        ones = torch.ones(n_samples, 1, device=self.device)
        X = torch.cat([states, ones], dim=1) # [n_samples, input_dim + 1]
        Y = targets
        
        # Ridge Regression Solution:
        # W = (X^T X + alpha I)^-1 X^T Y
        
        XT = X.t()
        XTX = torch.matmul(XT, X)
        
        # Regularization matrix (identity)
        # Don't regularize bias? Usually fine to regularize all or exclude bias.
        # Let's regularize all for simplicity.
        I = torch.eye(XTX.shape[0], device=self.device)
        
        # Solve
        # (XTX + alpha*I) W = XTY
        # W = ...
        
        # XTY: [input_dim+1, output_dim]
        XTY = torch.matmul(XT, Y)
        
        # LHS: [input_dim+1, input_dim+1]
        LHS = XTX + self.alpha * I
        
        # W: [input_dim+1, output_dim]
        # Use torch.linalg.solve or lstsq
        # solve expects AX = B. Here LHS * W = XTY
        try:
            W_aug = torch.linalg.solve(LHS, XTY)
        except RuntimeError:
            # Fallback to pseudoinverse if singular (unlikely with ridge)
            W_aug = torch.matmul(torch.linalg.pinv(LHS), XTY)
            
        # Extract weights and bias
        # W_aug = [W; b] (if we appended 1 at end)
        
        weights = W_aug[:-1, :] # [input_dim, output_dim]
        bias = W_aug[-1, :]     # [output_dim]
        
        # Update Linear layer
        with torch.no_grad():
            self.linear.weight.copy_(weights.t()) # nn.Linear stores as [out, in]
            self.linear.bias.copy_(bias)
            
        return self.linear.weight, self.linear.bias
