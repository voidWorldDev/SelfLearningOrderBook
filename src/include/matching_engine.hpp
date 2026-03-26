#pragma once
#include "order_book.hpp"
#include "order_flow.hpp"
#include <vector>
#include <deque>
#include <functional>

namespace lob {

// ─── MatchingEngine ───────────────────────────────────────────────────────────
// High-throughput event-driven matching engine.
// Accepts order events, routes them to the OrderBook, and broadcasts fills.
//
// Design: single-threaded core with callback-based event dispatch.
// For GPU-RL integration, the engine exposes a batch submission API.
class MatchingEngine {
public:
    using FillHandler   = std::function<void(const Trade&, uint32_t agent_id)>;
    using RejectHandler = std::function<void(OrderId, const std::string& reason)>;

    explicit MatchingEngine(OrderBook& book);

    // ── Event submission ──────────────────────────────────────────────────────
    void submit(Order order);

    // Batch submit (for RL agent actions, minimizes callback overhead)
    void submit_batch(std::vector<Order>& orders);

    // Process all queued events; returns number processed
    size_t process_events();

    // ── Callbacks ─────────────────────────────────────────────────────────────
    void on_fill(FillHandler h)   { fill_handler_   = std::move(h); }
    void on_reject(RejectHandler h){ reject_handler_ = std::move(h); }

    // ── Metrics ───────────────────────────────────────────────────────────────
    struct EngineStats {
        uint64_t total_submitted;
        uint64_t total_matched;
        uint64_t total_cancelled;
        uint64_t total_rejected;
        double   avg_queue_depth;    // average event queue depth
        double   throughput_ops_sec; // last measured throughput
    };
    EngineStats stats() const { return stats_; }
    void        reset_stats();

private:
    OrderBook&         book_;
    std::deque<Order>  event_queue_;
    FillHandler        fill_handler_;
    RejectHandler      reject_handler_;
    EngineStats        stats_ = {};

    void dispatch_fills(const std::vector<Trade>& trades);
    bool validate_order(const Order& o, std::string& err) const;
};

} // namespace lob
