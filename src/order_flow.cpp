#include "include/order_flow.hpp"
#include <cmath>
#include <algorithm>

namespace lob {

StochasticFlowGenerator::StochasticFlowGenerator(const OrderFlowConfig& cfg, uint64_t seed)
    : cfg_(cfg), rng_(seed) {}

std::optional<Order> StochasticFlowGenerator::next_order(
    std::optional<Price> best_bid,
    std::optional<Price> best_ask,
    Timestamp current_time)
{
    // Combined total arrival rate (Poisson superposition)
    double total_rate = cfg_.lambda_limit_buy  + cfg_.lambda_limit_sell
                      + cfg_.lambda_market_buy + cfg_.lambda_market_sell
                      + cfg_.lambda_cancel;

    // Probability each event type fires in this call (approximation: small dt)
    // Use uniform draw to select which stream fires
    std::uniform_real_distribution<double> udist(0.0, total_rate);
    double draw = udist(rng_);

    double cum = 0;
    OrderType otype;
    Side      oside;

    cum += cfg_.lambda_limit_buy;
    if (draw < cum) { otype = OrderType::Limit;  oside = Side::Buy;  goto make_order; }
    cum += cfg_.lambda_limit_sell;
    if (draw < cum) { otype = OrderType::Limit;  oside = Side::Sell; goto make_order; }
    cum += cfg_.lambda_market_buy;
    if (draw < cum) { otype = OrderType::Market; oside = Side::Buy;  goto make_order; }
    cum += cfg_.lambda_market_sell;
    if (draw < cum) { otype = OrderType::Market; oside = Side::Sell; goto make_order; }

    // Cancel — return nullopt (cancel is handled by engine referencing live order ids)
    return std::nullopt;

make_order:
    Price    price = (otype == OrderType::Limit)
                     ? sample_limit_price(oside, best_bid, best_ask)
                     : INVALID_PRICE;
    Quantity qty   = sample_qty();
    OrderId  id    = next_id_++;

    return Order(id, oside, otype, price, qty, current_time, 0 /*stochastic flow*/);
}

std::vector<Order> StochasticFlowGenerator::advance(
    std::optional<Price> best_bid,
    std::optional<Price> best_ask,
    uint64_t dt_ns)
{
    std::vector<Order> orders;
    double dt_s = dt_ns * 1e-9;
    double total_rate = cfg_.lambda_limit_buy  + cfg_.lambda_limit_sell
                      + cfg_.lambda_market_buy + cfg_.lambda_market_sell;

    // Expected number of orders in this interval
    std::poisson_distribution<int> pdist(total_rate * dt_s);
    int n = pdist(rng_);

    for (int i = 0; i < n; ++i) {
        clock_ns_ += dt_ns / std::max(n, 1);
        auto o = next_order(best_bid, best_ask, clock_ns_);
        if (o) orders.push_back(*o);
    }
    clock_ns_ += dt_ns;
    return orders;
}

Price StochasticFlowGenerator::sample_limit_price(
    Side side,
    std::optional<Price> best_bid,
    std::optional<Price> best_ask)
{
    // Fallback to initial mid if book is empty
    Price reference;
    if (side == Side::Buy) {
        reference = best_bid.value_or(static_cast<Price>(cfg_.init_mid_price - cfg_.tick_size));
    } else {
        reference = best_ask.value_or(static_cast<Price>(cfg_.init_mid_price + cfg_.tick_size));
    }

    // Geometric distribution: ticks from best quote
    std::geometric_distribution<int> gdist(cfg_.price_level_decay);
    int ticks = std::min(gdist(rng_), cfg_.max_ticks_from_best);

    Price tick = static_cast<Price>(cfg_.tick_size);
    if (side == Side::Buy) {
        return reference - ticks * tick;
    } else {
        return reference + ticks * tick;
    }
}

Quantity StochasticFlowGenerator::sample_qty() {
    std::lognormal_distribution<double> ldist(cfg_.qty_mean, cfg_.qty_sigma);
    double raw = ldist(rng_);
    return static_cast<Quantity>(std::max(1.0, std::round(raw)));
}

} // namespace lob
