#include "include/simulator.hpp"
#include "include/logger.hpp"
#include <iostream>
#include <algorithm>

namespace lob {

Simulator::Simulator(const SimConfig& cfg)
    : cfg_(cfg),
      engine_(book_),
      flow_gen_(cfg.flow_cfg),
      spread_metrics_(1000),
      fill_estimator_(10),
      inv_tracker_(100)
{
    // Wire fill callback
    engine_.on_fill([this](const Trade& t, uint32_t agent_id) {
        on_fill(t, agent_id);
    });
}

void Simulator::add_agent(std::shared_ptr<RLAgent> agent) {
    agents_.push_back(std::move(agent));
}

SimResult Simulator::run() {
    Logger::get().info("Simulation starting: duration=",
                       cfg_.duration_ns / 1'000'000'000ULL, "s");

    uint64_t last_snapshot = 0;
    uint64_t t = 0;

    while (t < cfg_.duration_ns) {
        tick(t);
        t += cfg_.tick_dt_ns;

        // Book snapshot at interval
        if (cfg_.log_book_snapshot && t - last_snapshot >= cfg_.snapshot_interval_ns) {
            record_snapshot(t);
            last_snapshot = t;
        }
    }

    // Collect results
    SimResult r{};
    r.total_trades  = book_.trade_count();
    r.total_orders  = book_.order_count();
    r.realized_vol  = spread_metrics_.realized_vol();
    r.mean_spread_ticks = spread_metrics_.mean_spread();
    r.mean_fill_prob = fill_estimator_.fill_prob(1); // fill prob at 1 tick
    r.agent_inventory= inv_tracker_.state();
    r.agent_pnl     = inv_tracker_.state().realized_pnl
                    + inv_tracker_.state().unrealized_pnl;
    r.trade_log     = trade_log_;

    Logger::get().info("Simulation complete. Trades=", r.total_trades,
                       " Orders=", r.total_orders,
                       " AgentPnL=", r.agent_pnl);
    return r;
}

void Simulator::tick(uint64_t t_ns) {
    // 1. Generate background stochastic flow
    auto bg_orders = flow_gen_.advance(book_.best_bid(), book_.best_ask(),
                                       cfg_.tick_dt_ns);
    engine_.submit_batch(bg_orders);

    // 2. Agent actions
    agent_step(t_ns);

    // 3. Process all queued events
    engine_.process_events();

    // 4. Update spread metrics
    if (book_.best_bid() && book_.best_ask()) {
        SpreadSnapshot snap;
        snap.ts          = t_ns;
        snap.best_bid    = *book_.best_bid();
        snap.best_ask    = *book_.best_ask();
        snap.spread      = *book_.spread();
        snap.mid         = *book_.mid_price();
        snap.bid_top_qty = book_.bid_qty_at(snap.best_bid);
        snap.ask_top_qty = book_.ask_qty_at(snap.best_ask);
        spread_metrics_.record(snap);
        inv_tracker_.update_mid(snap.mid);
    }
}

void Simulator::agent_step(uint64_t t_ns) {
    if (agents_.empty()) return;

    double time_remaining_s = static_cast<double>(cfg_.duration_ns - t_ns) * 1e-9;
    StateVec state = build_state(book_, inv_tracker_.state(),
                                 spread_metrics_, time_remaining_s);

    for (auto& agent : agents_) {
        ActionVec action = agent->act(state);

        // Convert (bid_offset, ask_offset) in ticks to quotes
        auto bb = book_.best_bid();
        auto ba = book_.best_ask();
        if (!bb || !ba) continue;

        Price tick = static_cast<Price>(cfg_.flow_cfg.tick_size);
        Price bid_price = *bb - static_cast<Price>(action[0]) * tick;
        Price ask_price = *ba + static_cast<Price>(action[1]) * tick;

        if (bid_price >= ask_price || bid_price <= 0) continue;

        static uint64_t agent_order_id = 1'000'000'000ULL;

        Order bid(agent_order_id++, Side::Buy,  OrderType::Limit,
                  bid_price, 10, t_ns, agent->agent_id());
        Order ask(agent_order_id++, Side::Sell, OrderType::Limit,
                  ask_price, 10, t_ns, agent->agent_id());

        engine_.submit(bid);
        engine_.submit(ask);
    }
}

void Simulator::record_snapshot(uint64_t t_ns) {
    std::cout << "\n─── Book Snapshot @ " << t_ns / 1'000'000 << " ms ───\n";
    print_book(book_, 5);
}

void Simulator::on_fill(const Trade& t, uint32_t agent_id) {
    if (cfg_.log_trades) {
        trade_log_.push_back(t);
    }

    // Track inventory for agent fills
    if (agent_id > 0) {
        inv_tracker_.on_fill(t, t.aggressor_side);
    }

    fill_estimator_.record_fill(t.passive_id, t.timestamp);
    fill_estimator_.record_fill(t.aggressor_id, t.timestamp);
}

} // namespace lob
