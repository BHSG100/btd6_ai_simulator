# BTD6 AI Simulator (v2.1)

Simulador tático de alta performance inspirado em **Bloons TD 6**, construído em Python com **Numba JIT**, **NumPy (Structure of Arrays - SoA)**, **Gymnasium** e **Pygame**. O projeto foi desenhado para treinar e avaliar agentes autônomos via **Algoritmos Genéticos Adaptativos (GA)** e **Aprendizado por Reforço (MaskablePPO)** em múltiplas arenas simultâneas.

Repositorio oficial: [BHSG100/btd6_ai_simulator](https://github.com/BHSG100/btd6_ai_simulator)

## Destaques da Arquitetura (v2.0 / v2.1)

- **Motor Físico Macro-Step em Numba (`core/engine.py`)**: Simulação vetorizada em *Structure of Arrays (SoA)* compilada com `@njit(fastmath=True, nogil=True, cache=True)` através da função `step_macro_ticks_jit`. Suporta pulo de camadas (*layer skipping* via `apply_damage_with_skip`), tipos de dano por bitmask (`DMG_SHARP`, `DMG_EXPLOSION`, `DMG_ENERGY`, `DMG_FREEZE`, `DMG_NORMAL`), propriedades especiais de Bloons (`FLAG_CAMO`, `FLAG_LEAD`, `FLAG_BLACK`, `FLAG_PINK`, `FLAG_FORTIFIED`, `FLAG_REGROW`), modos de mira (`TARGET_FIRST`, `TARGET_LAST`, `TARGET_CLOSE`, `TARGET_STRONG`), dano em área (splash) e auras de suporte (Monkey Village e Alchemist).
- **Ambiente Tático & Crosspathing (`ai/smart_env.py`)**:
  - **`BTD6TacticalEnv` (Gymnasium)**: Ambiente compatível com RL e evolução genética, suportando as rodadas 1 a 100 (incluindo MOABs, BFBs, ZOMGs, DDTs e o B.A.D. na rodada 100), modo **CHIMPS**, até 512 Bloons simultâneos, 24 âncoras táticas de posicionamento (`compute_tactical_anchors`) e catálogo tático (`TOWER_CATALOG`) com 8 arquétipos (Dart `0-2-3`, Bomb `2-0-4`, Ninja `4-0-2`, Alch `4-2-0`, Super `2-0-3`, Village `2-3-0`, Farm `2-3-0` e Sniper `4-0-2`).
  - **`SmartEnv`**: Gerenciamento estrito de upgrades com validação de **Crosspathing 5-2-0** (dois caminhos ativos, terceiro bloqueado, limite secundário no Tier 2 e trava global de Tier 5 único por tipo de torre) e venda de torres com reembolso de 70% (`sell_tower`).
- **Otimizador Genético Adaptativo (`ai/hybrid_trainer.py`)**: Implementa a classe `AdaptiveGeneticOptimizer` com elitismo, crossover uniforme, hipermutação adaptativa e persistência automática do melhor genoma em `memory/best_genome.json`.
- **Cyber Arena HUD 60 FPS (`ui/visualizer.py`)**: Visualizador multi-arena em tempo real (`CyberArenaVisualizer`) construído com `pygame` e `pygame.gfxdraw`, exibindo renderização suavizada (anti-aliasing), buffers de feixes/lasers por tipo de dano, mapa de calor e telemetria das gerações.

## Estrutura do Repositório

### Pipeline Ativo (v2.1)

| Arquivo | Descrição |
| :--- | :--- |
| [`main.py`](https://github.com/BHSG100/btd6_ai_simulator/blob/main/main.py) | Ponto de entrada principal (`run_visual_training`) que integra múltiplas arenas `BTD6TacticalEnv`, o `AdaptiveGeneticOptimizer` e o `CyberArenaVisualizer`. |
| [`core/engine.py`](https://github.com/BHSG100/btd6_ai_simulator/blob/main/core/engine.py) | Kernel Numba JIT (`step_macro_ticks_jit`), física de dano com *layer skipping* (`apply_damage_with_skip`), flags de imunidade e geração de Bloons por rodada (`spawn_bloon_for_round`). |
| [`ai/smart_env.py`](https://github.com/BHSG100/btd6_ai_simulator/blob/main/ai/smart_env.py) | Ambientes `BTD6TacticalEnv` e `SmartEnv`, cálculo de chokepoints (`compute_tactical_anchors`) e catálogo de torres (`TOWER_CATALOG`). |
| [`ai/hybrid_trainer.py`](https://github.com/BHSG100/btd6_ai_simulator/blob/main/ai/hybrid_trainer.py) | Treinador `AdaptiveGeneticOptimizer` com crossover, mutação adaptativa e salvamento em `memory/best_genome.json`. |
| [`ui/visualizer.py`](https://github.com/BHSG100/btd6_ai_simulator/blob/main/ui/visualizer.py) | Interface gráfica `CyberArenaVisualizer` em Pygame (60 FPS) para acompanhar múltiplas arenas em paralelo. |
| [memory/best\_genome.json](https://www.google.com/search?q=https://github.com/BHSG100/btd6_ai_simulator/blob/main/memory/best_genome.json) | Checkpoint JSON do melhor genoma evoluído pelo treinador híbrido na arquitetura v2.x. |

### Dados, Utilitários e Módulos Legados (v1)

| Caminho | Status | Descrição |
| :--- | :--- | :--- |
| [data/](https://www.google.com/search?q=https://github.com/BHSG100/btd6_ai_simulator/tree/main/data) | Suporte / Dados | Schemas JSON de Bloons (`bloons.json`), configurações (`game_config.json`), rodadas (`rounds.json`) e torres (`towers.json`). |
| [utils/](https://www.google.com/search?q=https://github.com/BHSG100/btd6_ai_simulator/tree/main/utils) | Utilitários v1 | Carregador Pydantic (`data_loader.py`), limites de hardware (`hardware_manager.py`) e cofre de memória v1 (`memory_vault.py`). |
| `core/arenas.py`, `grid.py`, `jit_kernels.py`, `path.py` | Legado (v1) | Antiga estrutura de arenas (`BloonArena`, `TowerArena`), grid 2D e kernels JIT da versão v1. |
| `ai/gym_env.py`, `ga_evaluator.py`, `evaluate_agent.py`, `train_ga.py`, `train_ppo.py`, `train_ppo_gpu.py` | Legado (v1) | Scripts de treinamento e avaliação da v1 que dependiam da antiga classe `GameEngine` ou do módulo `core.simulator`. |
| `benchmark.py`, `btd6_control_center.py`, `visual_watcher.py` | Legado (v1) | Scripts monolíticos de benchmark e visualização da primeira versão. |

## Instalação e Requisitos

1. Clone o repositório:

```bash
git clone https://github.com/BHSG100/btd6_ai_simulator.git
cd btd6_ai_simulator
```

2. Crie e ative um ambiente virtual (Windows PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Instale as dependências:

```powershell
pip install -r requirements.txt
```

*(Principais pacotes utilizados: `numpy`, `numba`, `gymnasium`, `pygame`, `pydantic`, `torch`, `sb3-contrib`, `stable-baselines3`).*

## Como Executar

Para iniciar o treinamento visual multi-arena com o Algoritmo Genético Adaptativo e o Cyber HUD a 60 FPS:

```powershell
python main.py
```

O script inicializa as arenas táticas (`BTD6TacticalEnv`), carrega ou cria a população no `AdaptiveGeneticOptimizer`, executa os macro-ticks compilados via Numba (`step_macro_ticks_jit`) e salva automaticamente os melhores genomas em `memory/best_genome.json`.

## Limpeza Recomendada (Remoção de Código Morto da v1)

Para manter o repositório enxuto e alinhado exclusivamente com a arquitetura v2.1, execute os comandos abaixo na raiz do projeto para remover os módulos legados da v1 e artefatos binários antigos:

```powershell
git rm benchmark.py btd6_control_center.py visual_watcher.py
git rm core/arenas.py core/grid.py core/jit_kernels.py core/path.py
git rm ai/evaluate_agent.py ai/ga_evaluator.py ai/gym_env.py ai/train_ga.py ai/train_ppo.py ai/train_ppo_gpu.py
git rm -r checkpoints/
git rm memory/btd6_meta_chimps.json memory/btd6_meta_easy.json memory/btd6_meta_hard.json memory/btd6_meta_medium.json
```
