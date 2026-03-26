#pragma once
#include "order.hpp"
#include <random>
#include <cstdint>
#include <optional>

namespace lob {

// ─── OrderFlowConfig ─────────────────────────────────────────────────────────
// Parameters controlling the stochastic order flow process.
// Based on a simplified Cont (2010) model:
//   - Poisson arrivals for limit, market, cancel
//   - Geometric/log-normal quantity distribution
//   - Orders cluster near the spread
struct OrderFlowConfig {
    // Arrival rates (orders per second)
    double lambda_limit_buy  = 10.0;
    double lambda_limit_sell = 10.0;
    double lambda_market_buy  = 3.0;
    double lambda_market_sell = 3.0;
    double lambda_cancel      = 5.0;

    // Price placement: how far from best quote (in ticks), geometric distribution
    double price_level_decay = 0.5;   // higher = more orders near spread
    int    max_ticks_from_best = 20;

    // Quantity distribution (log-normal)
    double qty_mean  = 4.5;   // mean of ln(qty)
    double qty_sigma = 0.8;

    // Initial mid-price and tick size
    double init_mid_price = 10000.0; // $100.00
    double tick_size      = 1.0;     // $0.01 per tick
};

// ─── StochasticFlowGenerator ─────────────────────────────────────────────────
// Generates a stream of synthetic orders following the configured stochastic
// process. Thread-safe with its own PRNG state.
class StochasticFlowGenerator {
public:
    explicit StochasticFlowGenerator(const OrderFlowConfig& cfg = {},
                                     uint64_t seed = 42);

    // Generate the next order given the current best bid/ask
    // Returns nullopt if no order fires in this tick
    std::optional<Order> next_order(
        std::optional<Price> best_bid,
        std::optional<Price> best_ask,
        Timestamp current_time);

    // Advance internal clock by dt_ns nanoseconds; may emit multiple orders
    std::vector<Order> advance(
        std::optional<Price> best_bid,
        std::optional<Price> best_ask,
        uint64_t dt_ns);

    const OrderFlowConfig& config() const { return cfg_; }
    void set_config(const OrderFlowConfig& c) { cfg_ = c; }

private:
    OrderFlowConfig       cfg_;
    std::mt19937_64       rng_;
    OrderId               next_id_   = 1;
    Timestamp             clock_ns_  = 0;

    Price sample_limit_price(Side side,
                             std::optional<Price> best_bid,
                             std::optional<Price> best_ask);
    Quantity sample_qty();
};

} // namespace lob
