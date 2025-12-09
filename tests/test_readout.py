import torch
import pytest
from pyngn.readout import ReadoutLayer

def test_readout_linear_regression():
    """Test if readout can learn a simple linear function y = 2x + 1."""
    input_dim = 1
    output_dim = 1
    readout = ReadoutLayer(input_dim, output_dim, alpha=0.0) # No regularization for exact fit
    
    # Training data
    # X = [[0], [1], [2], [3]]
    # Y = [[1], [3], [5], [7]]
    X = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    Y = torch.tensor([[1.0], [3.0], [5.0], [7.0]])
    
    # Fit
    readout.fit(X, Y)
    
    # Check weights
    # Weight should be 2.0, Bias should be 1.0
    assert torch.isclose(readout.linear.weight, torch.tensor([[2.0]]), atol=1e-5)
    assert torch.isclose(readout.linear.bias, torch.tensor([1.0]), atol=1e-5)
    
    # Predict
    # x=4 -> y=9
    pred = readout(torch.tensor([[4.0]]))
    assert torch.isclose(pred, torch.tensor([[9.0]]), atol=1e-5)

def test_readout_multidimensional():
    """Test multidimensional regression."""
    # y = x1 + 0.5*x2
    input_dim = 2
    output_dim = 1
    readout = ReadoutLayer(input_dim, output_dim, alpha=0.0)
    
    X = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 2.0]
    ])
    Y = torch.tensor([
        [1.0],
        [0.5],
        [1.5],
        [3.0]
    ])
    
    readout.fit(X, Y)
    
    # Weights: [1.0, 0.5], Bias: 0.0
    expected_weight = torch.tensor([[1.0, 0.5]])
    assert torch.allclose(readout.linear.weight, expected_weight, atol=1e-5)
    assert torch.isclose(readout.linear.bias, torch.tensor([0.0]), atol=1e-5)
