
# Especificação Formal: 3GSNN-LSM Architecture

## 1. Topologia do Sistema (O Hipergrafo)

O sistema é representado por um **Grafo Dinâmico Dirigido e Ponderado** $\mathcal{G}(t) = (\mathcal{V}, \mathcal{E}(t))$, imerso em um espaço métrico tridimensional $\mathbb{R}^3$.

### 1.1 Conjunto de Vértices ($\mathcal{V}$)
O conjunto total de nós é particionado em três subconjuntos funcionais:
$$ \mathcal{V} = \mathcal{V}_{in} \cup \mathcal{V}_{res} \cup \mathcal{V}_{out} $$

1.  **$\mathcal{V}_{in}$ (Input Layer):** $N_{in}$ neurônios transdutores virtuais.
2.  **$\mathcal{V}_{res}$ (Reservoir):** $N_{res}$ neurônios excitatórios e inibitórios (proporção típica 80/20) organizados esparsamente.
3.  **$\mathcal{V}_{out}$ (Readout):** $N_{out}$ unidades integradoras para decodificação.

### 1.2 Espaço de Estado ($\Psi$)
O estado global do sistema no instante $t$ é definido pela tupla:
$$ \Psi(t) = \langle \mathbf{u}(t), \mathbf{s}(t), \mathbf{M}(t), \mathbf{W}, \mathbf{D}(t), \mathbf{c}(t), \mathbf{\gamma}(t), \mathbf{H}(t) \rangle $$

*   **Variáveis Neurais (Rápidas):**
    *   $\mathbf{u} \in \mathbb{R}^{N_{res}}$: Potencial de membrana.
    *   $\mathbf{s} \in \{0, 1\}^{N_{res}}$: Vetor de spikes (disparos).
*   **Variáveis Sinápticas e Gliais (Lentas/Adaptativas):**
    *   $\mathbf{M} \in \{0, 1\}^{N_{res} \times N_{res}}$: Matriz de Adjacência (Máscara Microglial).
    *   $\mathbf{W} \in \mathbb{R}^{N_{res} \times N_{res}}$: Matriz de Pesos Sinápticos (Força base).
    *   $\mathbf{D} \in \mathbb{N}^{N_{res} \times N_{res}}$: Matriz de Atrasos (Oligodendrócitos).
    *   $\mathbf{c} \in \mathbb{R}^{N_{res}}$: Concentração de Cálcio Astrocitário.
    *   $\mathbf{\gamma} \in \mathbb{R}^{N_{res}}$: Ganho Astrocitário (Modulação).
    *   $\mathbf{H} \in [0, 1]^{N_{res} \times N_{res}}$: Saúde Sináptica.

---

## 2. Dinâmica Neural (Escala de Tempo: $dt \approx 1ms$)

A evolução do reservatório é governada pelo modelo **Leaky Integrate-and-Fire (LIF)** com injeção de corrente gliomórfica.

### 2.1 Equação Diferencial do Potencial
Para cada neurônio $i \in \mathcal{V}_{res}$:

$$ \tau_m \frac{du_i(t)}{dt} = -(u_i(t) - u_{rest}) + R \cdot I_{total}(i, t) $$

Onde:
*   $\tau_m$: Constante de tempo da membrana.
*   $u_{rest}$: Potencial de repouso.
*   $R$: Resistência de membrana (normalmente normalizada para 1).

### 2.2 Mecanismo de Disparo (Spiking)
$$
s_i(t) =
\begin{cases}
1 & \text{se } u_i(t) \geq \theta_{thresh} \text{ e } t > t_{last\_spike} + t_{ref} \\
0 & \text{caso contrário}
\end{cases}
$$
Após o disparo ($s_i(t)=1$), ocorre o reset: $u_i(t) \leftarrow u_{reset}$.

### 2.3 A Corrente Pentasináptica ($I_{total}$)
Esta equação integra a contribuição de todos os agentes. A corrente total recebida pelo neurônio $i$ é a soma da entrada externa e da recorrência interna modulada:

$$ I_{total}(i, t) = I_{ext}(i, t) + \underbrace{\gamma_i(t)}_{\text{Astrócito}} \cdot \sum_{j \in \mathcal{V}_{res}} \underbrace{M_{ij}(t)}_{\text{Microglia}} \cdot w_{ij} \cdot \underbrace{s_j(t - D_{ij}(t))}_{\text{Oligodendrócito}} $$

---

## 3. Dinâmicas Gliais (Escalas de Tempo: $10ms \to 1000s$)

Os componentes gliais não processam o sinal diretamente, mas modulam os parâmetros da equação pentasináptica.

### 3.1 Astrócitos: Homeostase de Ganho (Regulador PID Químico)
O astrócito monitora a taxa de disparo local e ajusta o ganho global ($\gamma$) do neurônio para manter o reservatório na "Borda do Caos" (regime crítico).

*   **Integração de Cálcio (Filtro Passa-Baixa):**
    $$ \tau_{Ca} \frac{dc_i(t)}{dt} = -c_i(t) + s_i(t) $$
    *(O nível de cálcio $c_i$ é uma média móvel da atividade elétrica).*

*   **Regra de Controle de Ganho ($\gamma$):**
    O objetivo é manter $c_i$ próximo de um *Set Point* ($\rho_{target}$).
    $$ \gamma_i(t+1) = \gamma_i(t) + \eta_{astro} \cdot (\rho_{target} - c_i(t)) $$
    *   **Lógica:** Se a atividade é baixa ($c < \rho$), o termo é positivo $\to$ Ganho aumenta. Se alta, Ganho diminui.
    *   **Restrição:** $\gamma_i \in [\gamma_{min}, \gamma_{max}]$.

### 3.2 Oligodendrócitos: Plasticidade de Atraso (Delay Learning)
Oligodendrócitos ajustam a velocidade de condução (atraso $D_{ij}$) para enriquecer a diversidade temporal do reservatório.

*   **Atualização Estocástica Baseada em Correlação:**
    A cada intervalo $T_{oligo}$:
    Seja $Corr(i, j)$ a correlação de disparos entre pré e pós-sináptico.
    $$ D_{ij}(t+1) = D_{ij}(t) + \Delta D $$
    $$
    \Delta D =
    \begin{cases}
    +1 & \text{se } P < p_{mut} \text{ (Drift aleatório para exploração)} \\
    -sgn(t_{spike}^i - (t_{spike}^j + D_{ij})) & \text{se } |Corr| > \theta_{sync} \text{ (Ajuste fino STDP)} \\
    0 & \text{caso contrário}
    \end{cases}
    $$
    *   **Objetivo:** Evitar que todos os sinais cheguem ao mesmo tempo (sincronia excessiva) ou aleatoriamente demais. Cria uma "memória de delay".

### 3.3 Microglia: Computação Morfológica (Poda e Manutenção)
A Microglia define a topologia ativa $\mathbf{M}$ baseada na eficiência energética.

*   **Variável de Saúde ($H_{ij}$):**
    Acumula "crédito" quando a sinapse transmite um spike útil (pré dispara e pós dispara logo depois).
    $$ H_{ij}(t) = (1-\beta)H_{ij}(t-1) + \beta \cdot (s_i(t) \cdot s_j(t - D_{ij})) - \lambda_{metabolica} $$
    *   $\lambda_{metabolica}$: Custo fixo de manter uma conexão viva.

*   **Regra de Poda (Pruning):**
    A cada intervalo $T_{micro}$:
    $$
    M_{ij}(t) =
    \begin{cases}
    0 & \text{se } H_{ij}(t) < \theta_{death} \text{ (Fagocitose)} \\
    1 & \text{se } H_{ij}(t) > \theta_{life} \\
    M_{ij}(t-1) & \text{caso contrário (Histerese)}
    \end{cases}
    $$

---

## 4. Readout e Aprendizado (A Interface)

A camada de saída não afeta a dinâmica interna do reservatório (em modo LSM padrão), apenas observa e interpreta.

*   **Vetor de Estado Líquido ($\mathbf{x}$):**
    Como $s(t)$ é discreto/esparso, o readout lê os traços sinápticos (correntes pós-sinápticas):
    $$ \tau_{read} \frac{d\mathbf{x}(t)}{dt} = -\mathbf{x}(t) + \mathbf{s}(t) $$

*   **Equação de Saída ($\mathbf{y}$):**
    $$ \mathbf{y}(t) = \mathbf{W}_{out} \cdot \mathbf{x}(t) $$

*   **Treinamento (Ridge Regression ou Delta Rule):**
    A matriz $\mathbf{W}_{out}$ é ajustada para minimizar o erro quadrático médio entre $\mathbf{y}(t)$ e o alvo $target(t)$.
    $$ \Delta \mathbf{W}_{out} = \eta \cdot (target(t) - \mathbf{y}(t)) \cdot \mathbf{x}(t)^T $$

