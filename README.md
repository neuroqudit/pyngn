# pyngn: Python Neuro-Glia Networks

**pyngn** (Python Neuro-Glia Networks) is a high-performance framework dedicated to **Gliomorphic Computing**, implementing the novel **3GSNN-LSM** (Triglial Spiking Neural Network - Liquid State Machine) architecture.

## What is 3GSNN-LSM?

Unlike traditional Spiking Neural Networks that rely solely on neuronal activity, the **3GSNN-LSM** model integrates a complex synaptic dynamic by orchestrating the interaction between fast-spiking neurons and three distinct glial agents:

*   **Astrocytes**: For homeostatic gain control and memory reverberation.
*   **Oligodendrocytes**: For adaptive temporal delays.
*   **Microglia**: For structural plasticity and topological pruning.

This bio-inspired synergy allows `pyngn` to generate self-organizing, energy-efficient reservoirs capable of complex temporal processing and continuous adaptation, bridging the gap between biological plausibility and computational efficiency without relying on backpropagation through time.

## Dynamics

The system is modeled as a **Dynamic Directed Weighted Graph** $\mathcal{G}(t)$ in 3D space.

### 1. Neural Dynamics (LIF)
The reservoir uses the **Leaky Integrate-and-Fire** model with a "Pentasynaptic Current" that integrates all glial contributions:

$$ \tau_m \frac{du_i(t)}{dt} = -(u_i(t) - u_{rest}) + R \cdot I_{total}(i, t) $$

$$ I_{total}(i, t) = I_{ext}(i, t) + \underbrace{\gamma_i(t)}_{\text{Astrocyte}} \cdot \sum_{j} \underbrace{M_{ij}(t)}_{\text{Microglia}} \cdot w_{ij} \cdot \underbrace{s_j(t - D_{ij}(t))}_{\text{Oligodendrocyte}} $$

### 2. Glial Dynamics
*   **Astrocytes (Homeostasis)**: Regulate gain $\gamma_i$ based on calcium integration to maintain the reservoir at the "Edge of Chaos".
    $$ \gamma_i(t+1) = \gamma_i(t) + \eta_{astro} \cdot (\rho_{target} - c_i(t)) $$
*   **Oligodendrocytes (Delays)**: Adjust conduction delays $D_{ij}$ to create temporal diversity and memory.
*   **Microglia (Pruning)**: Optimize topology $M_{ij}$ by pruning energy-inefficient synapses based on Hebbian-like health tracking.

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
