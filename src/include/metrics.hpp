#pragma once
#include "order.hpp"
#include <deque>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <numeric>
#include <optional>

namespace lob {

// ─── InventoryTracker ─────────────────────────────────────────────────────────
// Tracks a market maker's inventory position and associated PnL.
struct InventoryState {
    int64_t  position;       // net shares (+ long, - short)
    double   realized_pnl;
    double   unrealized_pnl;
    double   total_fees;
    uint64_t num_fills;
    uint64_t num_buys;
    uint64_t num_sells;
};

class InventoryTracker {
public:
    explicit InventoryTracker(int64_t max_position = 100,
                              double  fee_rate = 2e-4 /* 0.02% */);

    void on_fill(const Trade& t, Side aggressor_side);
    void update_mid(Price mid);

    const InventoryState& state()      const { return state_; }
    bool                  at_limit()   const;
    double                inventory_risk() const; // position × volatility × mid

    void reset();

private:
    InventoryState state_    = {};
    int64_t        max_pos_;
    double         fee_rate_;
    Price          last_mid_ = 0;
    double         avg_cost_ = 0.0;
};

// ─── SpreadMetrics ────────────────────────────────────────────────────────────
// Rolling statistics on bid-ask spread and queue dynamics.
struct SpreadSnapshot {
    Timestamp ts;
    Price     best_bid;
    Price     best_ask;
    Price     spread;       // in ticks
    Price     mid;
    Quantity  bid_top_qty;
    Quantity  ask_top_qty;
};

class SpreadMetrics {
public:
    explicit SpreadMetrics(size_t window = 1000);

    void record(const SpreadSnapshot& snap);

    double mean_spread()   const;
    double spread_vol()    const; // std dev
    double mean_mid()      const;
    double realized_vol()  const; // mid-price returns std dev

    const std::deque<SpreadSnapshot>& history() const { return history_; }

private:
    size_t                       window_;
    std::deque<SpreadSnapshot>   history_;
    std::deque<double>           mid_returns_;
};

// ─── FillProbabilityEstimator ─────────────────────────────────────────────────
// Empirical fill-probability curves: P(fill | ticks_from_best, time_horizon)
class FillProbabilityEstimator {
public:
    explicit FillProbabilityEstimator(int max_ticks = 10);

    void record_placement(OrderId id, int ticks_from_best, Timestamp placed_at);
    void record_fill(OrderId id, Timestamp filled_at);
    void record_cancel(OrderId id);

    // Estimated probability for a given tick distance
    double fill_prob(int ticks_from_best) const;
    double avg_fill_time_ms(int ticks_from_best) const;

private:
    struct Placement { int ticks; Timestamp placed_at; bool filled; uint64_t fill_latency_ns; };
    std::unordered_map<OrderId, Placement> open_;
    std::vector<Placement>                 closed_;
    int                                    max_ticks_;
};

// ─── AlphaDecayAnalyzer ───────────────────────────────────────────────────────
// Measures how quickly predictive signal decays after order placement.
struct AlphaObservation {
    Timestamp placed_at;
    Price     mid_at_placement;
    Price     mid_at_fill;
    Price     exec_price;
    Side      side;
    int       ticks_from_best;
};

class AlphaDecayAnalyzer {
public:
    void record(const AlphaObservation& obs) { obs_.push_back(obs); }

    // Average implementation shortfall in ticks
    double mean_shortfall() const;
    // Realized spread capture: exec_price relative to mid
    double mean_spread_capture() const;

    const std::vector<AlphaObservation>& observations() const { return obs_; }

private:
    std::vector<AlphaObservation> obs_;
};

} // namespace lob
