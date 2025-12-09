# pyngn: Python Neuro-Glia Networks

**pyngn** (Python Neuro-Glia Networks) is a high-performance framework dedicated to **Gliomorphic Computing**, implementing the novel **3GSNN-LSM** (Triglial Spiking Neural Network - Liquid State Machine) architecture.

## What is 3GSNN-LSM?

Unlike traditional Spiking Neural Networks that rely solely on neuronal activity, the **3GSNN-LSM** model integrates a complex synaptic dynamic by orchestrating the interaction between fast-spiking neurons and three distinct glial agents:

*   **Astrocytes**: For homeostatic gain control and memory reverberation.
*   **Oligodendrocytes**: For adaptive temporal delays.
*   **Microglia**: For structural plasticity and topological pruning.

This bio-inspired synergy allows `pyngn` to generate self-organizing, energy-efficient reservoirs capable of complex temporal processing and continuous adaptation, bridging the gap between biological plausibility and computational efficiency without relying on backpropagation through time.

## Project Structure

```text
pyngn-project/
├── pyngn/                  # Core System Source Code
│   ├── __init__.py
│   ├── neuron.py           # LIF Dynamics and Base Tensors
│   ├── glia.py             # Glial Controllers (Astro, Oligo, Micro)
│   ├── synapse.py          # Weight Management and Delay Buffers
│   ├── reservoir.py        # Orchestrator Class (3GSNN)
│   └── readout.py          # Readout Layer (Ridge Regression/Delta)
├── tests/                  # Unit Tests (pytest)
├── notebooks/              # Experimentation and Benchmarks
├── pyproject.toml          # Build Configuration
└── README.md
```

## Installation

```bash
pip install pyngn
```

## Basic Usage

Here is a simple example demonstrating a layer of LIF neurons with synaptic delays (Oligodendrocytes):

```python
import torch
from pyngn.neuron import LIFLayer
from pyngn.synapse import DelayBuffer

# 1. Initialize Layers
n_neurons = 5
# Create a layer of 5 LIF neurons
layer = LIFLayer(n_neurons=n_neurons, tau_m=20.0, v_th=1.0)
# Create a delay buffer for these neurons
buffer = DelayBuffer(n_neurons=n_neurons, max_delay=10)

# 2. Define Connectivity and Delays
# Example: 5 neurons, each connected to others with specific delays
# Here we simulate a dummy delay matrix for demonstration
delays = torch.randint(1, 10, (n_neurons, n_neurons))

# 3. Simulation Loop
for t in range(100):
    # Retrieve delayed spikes from the past based on the delay matrix
    delayed_spikes = buffer.get_delayed_spikes(delays)
    
    # Calculate input current
    # Summing spikes from pre-synaptic neurons (simplified weight=1.0)
    i_syn = delayed_spikes.sum(dim=0) 
    
    # Update neuron state
    spikes = layer.forward(i_syn=i_syn)
    
    # Push new spikes to buffer
    buffer.push(spikes)
    
    if t % 10 == 0:
        print(f"Time {t}: Active Neurons {spikes.sum().item()}")
```
