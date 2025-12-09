Este é o **Plano Mestre de Implementação do `pyngn`**.

Este documento estrutura o desenvolvimento do pacote `pyngn` (Python Neuro-Glia Networks), focado na arquitetura **3GSNN-LSM** (Triglial Spiking Neural Networks - Liquid State Machine). O plano utiliza uma abordagem de desenvolvimento iterativo e incremental, onde cada fase resulta em um incremento funcional de software versionado.

Utilizaremos o template de projeto **Antigravity** (ou estrutura padrão similar de Python moderno: `pyproject.toml`, `src/`, etc.) para garantir boas práticas de empacotamento.

---

# Estrutura do Projeto

```text
pyngn-project/
│
├── pyngn/                  # Core do Sistema (Source Code)
│   ├── __init__.py
│   ├── neuron.py           # Dinâmica LIF e Tensores Base
│   ├── glia.py             # Controladores (Astro, Oligo, Micro)
│   ├── synapse.py          # Gerenciamento de Pesos e Buffers
│   ├── reservoir.py        # A classe orquestradora (3GSNN)
│   └── readout.py          # Camada de Saída (Ridge Regression/Delta)
│
├── tests/                  # Testes Unitários (pytest)
│   ├── test_neuron.py
│   ├── test_glia.py
│   ├── test_integration.py
│   └── conftest.py
│
├── notebooks/              # Experimentação e Benchmarks
│   ├── 01_lif_dynamics.ipynb
│   ├── 02_delayed_memory.ipynb
│   ├── 03_homeostasis_demo.ipynb
│   └── 04_benchmark_mnist.ipynb
│
├── pyproject.toml          # Configuração de Build/Dependencies
└── README.md
```

---

# Dataset de Referência e Benchmark

Para validar a **3GSNN-LSM**, utilizaremos o **Sequential MNIST (sMNIST)**.
*   **Por que sMNIST?** No MNIST tradicional, a imagem inteira é apresentada de uma vez. No *Sequential*, apresentamos a imagem pixel por pixel (ou linha por linha) como uma série temporal.
*   **Desafio:** A rede precisa ter **memória de curto prazo** para "lembrar" dos pixels do topo da imagem quando estiver lendo os de baixo para classificar o dígito. Isso valida a função dos Oligodendrócitos (Delays) e a reverberação do Reservatório.
*   **Comparativo:** Compararemos a acurácia e a eficiência (esparsidade) contra uma **RNN (Recurrent Neural Network)** simples ou **LSTM** implementada em PyTorch tradicional.

---

# Fases de Implementação

## Fase 1: O Motor Neural (Bare Metal Spiking)
**Versão Alvo:** `pyngn-v0.1.0`

### Objetivos
Implementar a dinâmica vetorializada de neurônios LIF e a integração sináptica básica em PyTorch, sem glia complexa ainda. Estabelecer o grafo computacional básico.

### Equações (Dinâmica Neural)
1.  **Potencial de Membrana ($\mathbf{u}$):**
    $$ \mathbf{u}[t+1] = \alpha \cdot \mathbf{u}[t] \cdot (1 - \mathbf{s}[t]) + \mathbf{I}_{syn}[t] + \mathbf{I}_{ext}[t] $$
    *   $\alpha = \exp(-dt/\tau_m)$: Fator de decaimento.
    *   Termo $(1 - \mathbf{s}[t])$: Implementa o *Hard Reset* após o disparo.
2.  **Spike ($\mathbf{s}$):**
    $$ \mathbf{s}[t] = \Theta(\mathbf{u}[t] - v_{th}) $$
    *   $\Theta$: Função Heaviside (Degrau).

### Implementação (`pyngn/neuron.py`)
*   Classe `LIFLayer(n_neurons, tau, v_th)`.
*   Uso de `torch.Tensor` para todos os estados.
*   Suporte a GPU (`cuda`).

### Testes Unitários (`tests/test_neuron.py`)
*   **`test_decay`**: Verificar se a voltagem decai exponencialmente sem entrada.
*   **`test_fire_and_reset`**: Injetar corrente constante e verificar se gera um trem de pulsos regular e se a voltagem zera após o disparo.
*   **`test_refractory`**: Garantir que o neurônio não dispara durante o período refratário.

### Notebooks
*   **`01_lif_dynamics.ipynb`**: Visualizar o potencial de membrana $u(t)$ e o raster plot de spikes de um pequeno grupo de neurônios.

---

## Fase 2: A Dimensão Temporal (Oligodendrócitos)
**Versão Alvo:** `pyngn-v0.2.0`

### Objetivos
Implementar o **Ring Buffer** (Buffer Circular) para gerenciar atrasos sinápticos heterogêneos. Esta é a base da memória do sistema.

### Equações (Dinâmica de Atraso)
1.  **Buffer de Spikes ($\mathbf{B}$):** Matriz de dimensão $[T_{max\_delay}, N_{res}]$.
2.  **Recuperação de Spikes Atrasados:**
    A corrente que chega no neurônio $i$ vinda de $j$ no tempo $t$ é baseada no spike que ocorreu em $t - D_{ij}$.
    $$ \mathbf{S}_{in}[j, i] = \mathbf{B}[(t - \mathbf{D}[j, i]) \% T_{max}, j] $$

### Implementação (`pyngn/synapse.py`)
*   Classe `DelayBuffer(max_delay, n_neurons)`.
*   Operações avançadas de indexação (`torch.gather`) para extrair os spikes corretos baseados na matriz de inteiros $\mathbf{D}$.

### Testes Unitários (`tests/test_integration.py`)
*   **`test_delay_consistency`**: Disparar neurônio A no tempo $t=0$ com conexão $D=5$ para neurônio B. Verificar se B recebe corrente *apenas* em $t=5$.
*   **`test_buffer_overwrite`**: Verificar se o buffer circular sobrescreve corretamente dados antigos sem corromper o histórico recente.

### Notebooks
*   **`02_delayed_memory.ipynb`**: Demonstrar como a variação de delays permite que a rede detecte padrões temporais (ex: detectar a sequência A-B-C com intervalos de tempo específicos).

---

## Fase 3: Homeostase e Estabilidade (Astrócitos)
**Versão Alvo:** `pyngn-v0.3.0`

### Objetivos
Implementar o controlador PID químico para regulação de ganho, permitindo que o reservatório mantenha atividade autossustentada (Criticalidade).

### Equações (Dinâmica Astrocitária)
1.  **Integração de Cálcio ($Ca$):**
    $$ \mathbf{Ca}[t+1] = \beta_{Ca} \cdot \mathbf{Ca}[t] + \mathbf{s}[t] $$
2.  **Controle de Ganho ($\gamma$):**
    $$ \mathbf{\gamma}[t+1] = \mathbf{\gamma}[t] + \eta_{astro} \cdot (\text{TargetRate} - \mathbf{Ca}[t]) $$
    *   Restrição: $\mathbf{\gamma} \in [\gamma_{min}, \gamma_{max}]$.
3.  **Modulação de Corrente:**
    $$ \mathbf{I}_{rec}[t] = (\mathbf{W} \cdot \mathbf{S}_{delayed}) \odot \mathbf{\gamma}[t] $$

### Implementação (`pyngn/glia.py`)
*   Classe `AstrocyteController`.
*   Integração no loop principal do reservatório.

### Testes Unitários (`tests/test_glia.py`)
*   **`test_gain_up`**: Se a entrada for zero, verificar se $\gamma$ sobe até o máximo.
*   **`test_gain_down`**: Se a entrada for saturante, verificar se $\gamma$ desce para atenuar a atividade.

### Notebooks
*   **`03_homeostasis_demo.ipynb`**: Iniciar o reservatório com atividade aleatória e remover o input. Mostrar como os astrócitos mantêm a "reverberação" (memória de trabalho) ativa por mais tempo que uma rede sem glia.

---

## Fase 4: O Arquiteto (Microglia) e Readout
**Versão Alvo:** `pyngn-v0.4.0`

### Objetivos
Implementar a poda estrutural (Máscara $\mathbf{M}$) para otimização e a camada de leitura (Readout) para realizar tarefas.

### Equações (Dinâmica Microglial)
1.  **Saúde Sináptica ($H$):**
    $$ H_{ij}[t+1] = H_{ij}[t] - \lambda_{decay} + \epsilon \cdot (\text{Pré}[t-D] \times \text{Pós}[t]) $$
    *(Hebbian-like reinforcement)*.
2.  **Poda (Topologia):**
    $$ M_{ij} = 0 \quad \text{se } H_{ij} < H_{death} $$

### Equações (Readout - Ridge Regression)
Para treinamento rápido (sem backprop no tempo):
$$ \mathbf{W}_{out} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{Y}_{target} $$
Onde $\mathbf{X}$ é a matriz de estados do reservatório acumulados ao longo do tempo.

### Implementação (`pyngn/glia.py`, `pyngn/readout.py`)
*   Classe `MicrogliaController`.
*   Classe `ReadoutLayer` com método `fit(states, targets)`.

### Testes Unitários
*   **`test_pruning`**: Criar uma conexão inútil e verificar se $\mathbf{M}_{ij}$ vira 0 após $N$ passos.
*   **`test_readout_convergence`**: Treinar a rede para resolver uma porta lógica XOR (com delay) e verificar erro zero.

---

## Fase 5: Integração Final e Benchmark (3GSNN-LSM)
**Versão Alvo:** `pyngn-v1.0.0`

### Objetivos
Consolidar todas as peças na classe `GHSNN` completa e rodar o benchmark principal.

### Implementação (`pyngn/reservoir.py`)
*   Classe `TriglialReservoir` que herda de `torch.nn.Module`.
*   Método `forward(inputs)` que executa o loop temporal e aciona as glias nas escalas de tempo corretas.

### Notebook Final
*   **`04_benchmark_mnist.ipynb`**:
    1.  Carregar **Sequential MNIST**.
    2.  Instanciar `pyngn.TriglialReservoir` (com ~1000 neurônios).
    3.  Rodar o dataset (processamento não-supervisionado no reservatório + treino supervisionado no readout).
    4.  Instanciar uma `torch.nn.LSTM` com número de parâmetros comparável.
    5.  **Comparar:** Acurácia final, tempo de convergência e robustez a ruído.

