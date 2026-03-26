#pragma once
#include "matching_engine.hpp"
#include "order_flow.hpp"
#include "rl_agent.hpp"
#include "metrics.hpp"
#include <memory>
#include <vector>
#include <string>

namespace lob {

// ─── SimConfig ────────────────────────────────────────────────────────────────
struct SimConfig {
    uint64_t  duration_ns       = 60ULL * 1'000'000'000ULL; // 60 seconds
    uint64_t  tick_dt_ns        = 1'000'000ULL;              // 1 ms ticks
    bool      log_trades        = true;
    bool      log_book_snapshot = false;
    uint64_t  snapshot_interval_ns = 1'000'000'000ULL;       // every 1 sec

    OrderFlowConfig flow_cfg;
    RewardConfig    reward_cfg;
};

// ─── SimResult ────────────────────────────────────────────────────────────────
struct SimResult {
    uint64_t          total_trades;
    uint64_t          total_orders;
    double            realized_vol;
    double            mean_spread_ticks;
    double            mean_fill_prob;
    InventoryState    agent_inventory;
    double            agent_pnl;
    std::vector<Trade> trade_log;
};

// ─── Simulator ────────────────────────────────────────────────────────────────
// Event-driven simulation loop integrating:
//   - Stochastic background order flow
//   - One or more RL agents quoting/executing
//   - Metrics collection
class Simulator {
public:
    explicit Simulator(const SimConfig& cfg = {});

    void add_agent(std::shared_ptr<RLAgent> agent);
    SimResult run();

    // Access internals for inspection
    const OrderBook&      book()    const { return book_; }
    const SpreadMetrics&  spreads() const { return spread_metrics_; }

private:
    SimConfig              cfg_;
    OrderBook              book_;
    MatchingEngine         engine_;
    StochasticFlowGenerator flow_gen_;
    SpreadMetrics          spread_metrics_;
    FillProbabilityEstimator fill_estimator_;
    AlphaDecayAnalyzer     alpha_analyzer_;
    InventoryTracker       inv_tracker_;

    std::vector<std::shared_ptr<RLAgent>> agents_;
    std::vector<Trade>                    trade_log_;

    // Per-tick handlers
    void   tick(uint64_t t_ns);
    void   agent_step(uint64_t t_ns);
    void   record_snapshot(uint64_t t_ns);

    // Callback wired to matching engine
    void on_fill(const Trade& t, uint32_t agent_id);
};

} // namespace lob
