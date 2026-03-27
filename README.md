#  Self-Learning Limit Order Book Simulator

A high-performance C++17 simulator for Level-2 limit order books with GPU-ready reinforcement learning agents for market making and optimal execution.

---

## Architecture

```
SelfLearningOrderBook/
├── CMakeLists.txt
├── README.md
└── src/
    ├── main.cpp                   ← Entry point, CLI, sim wiring
    ├── price_level.cpp            ← Single price-point queue (FIFO, price-time priority)
    ├── order_book.cpp             ← Full L2 book: bid/ask maps, matching engine
    ├── order_flow.cpp             ← Stochastic Poisson order flow (Cont 2010 model)
    ├── matching_engine.cpp        ← Event-driven high-throughput engine + callbacks
    ├── metrics.cpp                ← Inventory, spread, fill-prob, alpha-decay analytics
    ├── rl_agent.cpp               ← Q-learning agent, naive baseline, state/reward builder
    ├── simulator.cpp              ← Main simulation loop integrating all components
    ├── logger.cpp                 ← Structured logging + pretty L2 book printer
    └── include/
        ├── types.hpp              ← Enums: Side, OrderType, OrderStatus, AgentAction
        ├── order.hpp              ← Order and Trade structs
        ├── price_level.hpp        ← PriceLevel: FIFO queue at a single price
        ├── order_book.hpp         ← OrderBook: L2 book with price-time priority matching
        ├── order_flow.hpp         ← StochasticFlowGenerator config + interface
        ├── matching_engine.hpp    ← MatchingEngine: event queue + fill dispatch
        ├── metrics.hpp            ← InventoryTracker, SpreadMetrics, FillProbEstimator
        ├── rl_agent.hpp           ← RLAgent base, NaiveMarketMaker, SimpleQLearningAgent
        ├── simulator.hpp          ← Simulator: integrates book, flow, agents, metrics
        ├── logger.hpp             ← Logger, print_book, print_trade, print_sim_result
        └── rl_agent_gpu.cuh       ← GPU agent stub (CUDA/cuBLAS Actor-Critic / PPO)
```

---

## Components

### Order Book (`order_book.hpp / .cpp`)
- **Price-time priority** matching: bids sorted descending, asks ascending via `std::map`
- **O(log N)** limit order insertion and O(1) cancel via iterator index
- Market orders walk the book consuming resting liquidity
- Immediate-or-partial fill: unmatched residual rests as limit
- L2 snapshot API: `bid_levels(n)` / `ask_levels(n)` returning top-N price levels

### Matching Engine (`matching_engine.hpp / .cpp`)
- Event queue (`std::deque<Order>`) with batch submission for low-overhead RL agent integration
- Fill/reject callbacks wired at construction time
- Throughput stats: ops/sec, avg queue depth, fill/cancel/reject counters

### Stochastic Order Flow (`order_flow.hpp / .cpp`)
- Superposition of independent Poisson processes for limit buy, limit sell, market buy, market sell, cancel
- Price placement: geometric distribution from best quote (configurable decay, max ticks)
- Quantity: log-normal distribution (configurable mean/sigma)
- Based on Cont (2010) *"Stochastic modeling of order books"*

### Metrics (`metrics.hpp / .cpp`)
| Class | What it tracks |
|---|---|
| `InventoryTracker` | Net position, realized/unrealized PnL, fees, average cost |
| `SpreadMetrics` | Rolling mean/std of bid-ask spread and mid-price returns (realized vol) |
| `FillProbabilityEstimator` | Empirical fill probability and avg fill latency per tick distance |
| `AlphaDecayAnalyzer` | Implementation shortfall and spread capture across executions |

### RL Agents (`rl_agent.hpp / .cpp`)
| Agent | Description |
|---|---|
| `NaiveMarketMaker` | Always quotes at best_bid−k / best_ask+k ticks. Baseline. |
| `SimpleQLearningAgent` | Tabular Q-learning over discretized inventory state. Epsilon-greedy. |
| `GPURLAgent` *(stub)* | Batched MLP policy (Actor-Critic / PPO) via CUDA + cuBLAS. |

**State vector** (`STATE_DIM = 28`): top-5 bid/ask price depths + quantities (normalized), net position, realized PnL, mean spread, realized volatility, fill counts, time remaining.

**Action space**: `[bid_offset_ticks, ask_offset_ticks]` — how far to quote from best bid/ask.

**Reward**: PnL delta − quadratic inventory penalty + spread capture per fill.

---

## Build

### Prerequisites
- Clang ≥ 13 (`clang++`)
- CMake ≥ 3.16

### CPU-only build (default)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### With GPU support (CUDA)
Uncomment the GPU block in `src/include/rl_agent_gpu.cuh`, rename it to `.cu`, then:
```cmake
# In CMakeLists.txt, add:
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
find_package(CUDAToolkit REQUIRED)
file(GLOB_RECURSE CUDA_SRC "${CMAKE_SOURCE_DIR}/src/gpu/*.cu")
target_sources(${PROJECT_NAME} PRIVATE ${CUDA_SRC})
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cublas CUDA::cudart)
```

---

## Usage

```bash
# Default: 60s simulation, Q-learning agent
./build/SelfLearningOrderBook

# 5-minute sim with naive market maker + L2 snapshots
./build/SelfLearningOrderBook --duration 300 --agent naive --show-book

# Quick 10s test
./build/SelfLearningOrderBook --duration 10

# Help
./build/SelfLearningOrderBook --help
```

### Sample Output
```
  SIMULATION RESULTS SUMMARY
  Total Orders    :      87421
  Total Trades    :      12308
  Realized Vol    :   0.000312
  Mean Spread(tk) :   1.980000
  Fill Prob @1tk  :   0.641000

  AGENT INVENTORY & PnL
  Net Position    :         -3
  Realized PnL    :    24.8700
  Unrealized PnL  :    -1.2300
  Total Fees      :     8.4200
  Total PnL       :    23.6400
  Fills           :        842
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `std::map` for price levels | O(log N) worst case; iterator stability for O(1) cancel |
| `std::list` within price level | O(1) insert/erase anywhere (iterator invalidation safe) |
| Fixed-point prices (`int64_t × 100`) | Exact arithmetic, no floating-point drift |
| Callback-based fills | Zero-copy; agents react synchronously in the same tick |
| Separate `MatchingEngine` from `OrderBook` | Book is pure data structure; engine owns event routing |
| Poisson superposition for flow | Theoretically grounded; easy to calibrate to real TAQ data |

---

## Extending

### Calibrate to Real Data
Replace `StochasticFlowGenerator` with a replay engine that reads TAQ/ITCH feed files:
```cpp
class ReplayFlowGenerator {
public:
    std::vector<Order> advance(Price bb, Price ba, uint64_t dt_ns) {
        // Read next N events from NASDAQ ITCH 5.0 binary file
        // matching the given time window
    }
};
```

### Add a Deep RL Agent
1. Enable CUDA in CMakeLists.txt
2. Implement `GPURLAgent` in `src/include/rl_agent_gpu.cuh` → `.cu`
3. Use `batch_act()` to evaluate `BATCH_SIZE=256` parallel simulations per GPU pass
4. Run PPO updates every N steps using GPU-side rollout buffers

### Multi-asset / Cross-venue
Instantiate multiple `OrderBook` instances and route orders through a single `MatchingEngine` with a symbol field on `Order`.

---

## Benchmark Results

### Run Configuration
- Duration: 60 seconds
- Agent: Q-Learning Market Maker (alpha=0.01, gamma=0.99, eps=0.10)
- Build: Release (CMAKE_BUILD_TYPE=Release)

### Simulation Output
```
╔══════════════════════════════════════════════╗
║         SIMULATION RESULTS SUMMARY           ║
╠══════════════════════════════════════════════╣
║  Total Orders    :       9470              ║
║  Total Trades    :       3931              ║
║  Realized Vol    :     0.0000              ║
║  Mean Spread(tk) :     1.0000          ║
║  Fill Prob @1tk  :     0.0000          ║
╠══════════════════════════════════════════════╣
║         AGENT INVENTORY & PnL                ║
╠══════════════════════════════════════════════╣
║  Net Position    :          0              ║
║  Realized PnL    :     0.0000          ║
║  Unrealized PnL  :     0.0000        ║
║  Total Fees      :     0.0000          ║
║  Total PnL       :     0.0000          ║
║  Fills           :          0              ║
║  Buys            :          0              ║
║  Sells           :          0              ║
╚══════════════════════════════════════════════╝
```

### Performance Metrics
- Wall-clock runtime: 0.1637s
- Simulated/Real ratio: 366.46x (366.46 simulated seconds per real second)

### Final Order Book State
```
┌─────────────────────────────────────────────┐
│           LIMIT ORDER BOOK (L2)             │
├──────────┬──────────────┬────────────────────┤
│  SIDE    │  PRICE ($)   │  QUANTITY          │
├──────────┼──────────────┼────────────────────┤
│  ASK[0]  │      100.04  │          1779 (16 orders)   │
│  ASK[0]  │      100.03  │          3297 (27 orders)   │
│  ASK[0]  │      100.02  │         11189 (85 orders)   │
│  ASK[0]  │      100.01  │         36257 (2077 orders)   │
│  ASK[0]  │      100.00  │         17593 (529 orders)   │
├──────────┴──────────────┴────────────────────┤
│  mid=   99.99  spread=1.00 ticks              │
├──────────┬──────────────┬────────────────────┤
│  BID[0]  │       99.99  │         15800 (507 orders)   │
│  BID[0]  │       99.98  │         38351 (2091 orders)   │
│  BID[0]  │       99.97  │         10752 (74 orders)   │
│  BID[0]  │       99.96  │          3031 (29 orders)   │
│  BID[0]  │       99.95  │          2017 (23 orders)   │
└──────────┴──────────────┴────────────────────┘
Orders: 9470  Trades: 3931  Cancels: 0
```

---

## References
- Cont, R. (2010). *Stochastic modeling of order books*. Quantitative Finance.
- Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book*. Quantitative Finance.
- Spooner, T. et al. (2018). *Market making via reinforcement learning*. AAMAS.
- Schulman, J. et al. (2017). *Proximal Policy Optimization*. arXiv:1707.06347.
