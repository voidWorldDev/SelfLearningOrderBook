#pragma once
#include "order.hpp"
#include "order_book.hpp"
#include "metrics.hpp"
#include <array>
#include <vector>
#include <functional>

namespace lob {

// ─── Market State (observation space) ────────────────────────────────────────
// Feature vector fed to RL agents.
// Dimensionality: NUM_LEVELS*4 + 6 inventory/flow features.
static constexpr size_t NUM_LEVELS    = 5;
static constexpr size_t STATE_DIM     = NUM_LEVELS * 4 + 8;

using StateVec  = std::array<float, STATE_DIM>;
using ActionVec = std::array<float, 2>; // [bid_offset_ticks, ask_offset_ticks]

// Build a normalized state vector from current book + inventory
StateVec build_state(const OrderBook&        book,
                     const InventoryState&    inv,
                     const SpreadMetrics&     spread_metrics,
                     double                   time_remaining_s);

// ─── Agent interface ──────────────────────────────────────────────────────────
class RLAgent {
public:
    virtual ~RLAgent() = default;

    // Observe current state; return action (bid/ask quote offsets in ticks)
    virtual ActionVec act(const StateVec& state) = 0;

    // Receive reward signal after each step
    virtual void      observe_reward(float reward, bool done) = 0;

    // Persist / load learned policy
    virtual void save(const std::string& path) const {}
    virtual void load(const std::string& path) {}

    uint32_t agent_id() const { return agent_id_; }

protected:
    uint32_t agent_id_ = 0;
};

// ─── RewardFunction ───────────────────────────────────────────────────────────
// Configurable reward shaping for market making / optimal execution.
struct RewardConfig {
    double pnl_weight        = 1.0;
    double inventory_penalty = 0.01; // quadratic penalty on |position|
    double spread_reward     = 0.5;  // reward for captured half-spread
    double fill_reward       = 0.1;  // per fill
};

float compute_reward(const InventoryState& before,
                     const InventoryState& after,
                     const std::vector<Trade>& fills,
                     Price mid_price,
                     const RewardConfig& cfg);

// ─── Naive Market Maker (baseline heuristic agent) ────────────────────────────
// Always quotes at best_bid - k and best_ask + k ticks.
class NaiveMarketMaker : public RLAgent {
public:
    explicit NaiveMarketMaker(uint32_t id, int offset_ticks = 1);
    ActionVec act(const StateVec& state)            override;
    void      observe_reward(float, bool)           override {}

private:
    int offset_ticks_;
};

// ─── SimpleQLearningAgent ─────────────────────────────────────────────────────
// Tabular Q-learning over a discretized state-action space.
// CPU-only reference implementation; GPU version in gpu/rl_agent_gpu.cuh
class SimpleQLearningAgent : public RLAgent {
public:
    explicit SimpleQLearningAgent(uint32_t id,
                                  float alpha  = 0.01f,
                                  float gamma  = 0.99f,
                                  float eps    = 0.1f);

    ActionVec act(const StateVec& state) override;
    void      observe_reward(float reward, bool done) override;

    void save(const std::string& path) const override;
    void load(const std::string& path)       override;

private:
    float alpha_, gamma_, eps_;
    StateVec last_state_ = {};
    ActionVec last_action_ = {};
    bool has_prev_ = false;

    // Discretized Q-table (small grid for CPU demo)
    static constexpr int  BINS = 10;
    static constexpr int  ACTIONS = 9; // offsets {-1,0,1} × {-1,0,1}
    std::vector<float>    qtable_; // [state_bin * ACTIONS]

    int  discretize(const StateVec& s) const;
    int  greedy_action(int state_bin)  const;
    void update_q(int sb, int ab, float r, int sb_next);
};

} // namespace lob
