#include "include/matching_engine.hpp"
#include <chrono>
#include <sstream>

namespace lob {

MatchingEngine::MatchingEngine(OrderBook& book) : book_(book) {}

void MatchingEngine::submit(Order order) {
    event_queue_.push_back(std::move(order));
}

void MatchingEngine::submit_batch(std::vector<Order>& orders) {
    for (auto& o : orders) {
        event_queue_.push_back(std::move(o));
    }
}

size_t MatchingEngine::process_events() {
    size_t processed = 0;
    double total_queue = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    while (!event_queue_.empty()) {
        total_queue += static_cast<double>(event_queue_.size());
        Order order = event_queue_.front();
        event_queue_.pop_front();

        std::string err;
        if (!validate_order(order, err)) {
            ++stats_.total_rejected;
            if (reject_handler_) reject_handler_(order.id, err);
            ++processed;
            continue;
        }

        std::vector<Trade> trades;
        switch (order.type) {
            case OrderType::Limit:
                trades = book_.add_limit_order(std::move(order));
                break;
            case OrderType::Market:
                trades = book_.add_market_order(std::move(order));
                break;
            case OrderType::Cancel:
                book_.cancel_order(order.id);
                ++stats_.total_cancelled;
                break;
        }

        if (!trades.empty()) {
            ++stats_.total_matched;
            dispatch_fills(trades);
        }
        ++stats_.total_submitted;
        ++processed;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();
    if (elapsed_s > 0 && processed > 0) {
        stats_.throughput_ops_sec = static_cast<double>(processed) / elapsed_s;
    }
    if (processed > 0) {
        stats_.avg_queue_depth = total_queue / static_cast<double>(processed);
    }

    return processed;
}

void MatchingEngine::dispatch_fills(const std::vector<Trade>& trades) {
    if (!fill_handler_) return;
    for (auto& t : trades) {
        fill_handler_(t, 0 /*TODO: resolve agent_id from order_id*/);
    }
}

bool MatchingEngine::validate_order(const Order& o, std::string& err) const {
    if (o.qty == 0) {
        err = "zero quantity";
        return false;
    }
    if (o.type == OrderType::Limit && o.price <= 0) {
        err = "invalid limit price";
        return false;
    }
    return true;
}

void MatchingEngine::reset_stats() {
    stats_ = {};
}

} // namespace lob
