#include "include/order_book.hpp"
#include <stdexcept>

namespace lob {

// ─── Limit Order ──────────────────────────────────────────────────────────────
std::vector<Trade> OrderBook::add_limit_order(Order order) {
    ++order_count_;
    std::vector<Trade> trades;

    if (order.side == Side::Buy) {
        // Can immediately match against resting asks
        if (!asks_.empty()) {
            auto& [best_ask_price, best_level] = *asks_.begin();
            if (order.price >= best_ask_price) {
                trades = match_against(order, asks_);
            }
        }
    } else {
        // Can immediately match against resting bids
        if (!bids_.empty()) {
            auto& [best_bid_price, best_level] = *bids_.begin();
            if (order.price <= best_bid_price) {
                trades = match_against(order, bids_);
            }
        }
    }

    // If order has remaining quantity, insert as resting limit
    if (!order.is_filled() && order.remaining() > 0) {
        insert_limit(order);
    }
    return trades;
}

// ─── Market Order ─────────────────────────────────────────────────────────────
std::vector<Trade> OrderBook::add_market_order(Order order) {
    ++order_count_;
    if (order.side == Side::Buy) {
        return match_against(order, asks_);
    } else {
        return match_against(order, bids_);
    }
}

// ─── Cancel Order ─────────────────────────────────────────────────────────────
bool OrderBook::cancel_order(OrderId id) {
    auto it = order_index_.find(id);
    if (it == order_index_.end()) return false;

    auto& meta = it->second;
    if (meta.side == Side::Buy) {
        auto level_it = bids_.find(meta.price);
        if (level_it != bids_.end()) {
            level_it->second.remove(meta.it);
            if (level_it->second.empty()) bids_.erase(level_it);
        }
    } else {
        auto level_it = asks_.find(meta.price);
        if (level_it != asks_.end()) {
            level_it->second.remove(meta.it);
            if (level_it->second.empty()) asks_.erase(level_it);
        }
    }
    order_index_.erase(it);
    ++cancel_count_;
    return true;
}

// ─── Book Queries ─────────────────────────────────────────────────────────────
std::optional<Price> OrderBook::best_bid() const {
    if (bids_.empty()) return std::nullopt;
    return bids_.begin()->first;
}

std::optional<Price> OrderBook::best_ask() const {
    if (asks_.empty()) return std::nullopt;
    return asks_.begin()->first;
}

std::optional<Price> OrderBook::mid_price() const {
    auto bb = best_bid();
    auto ba = best_ask();
    if (!bb || !ba) return std::nullopt;
    return (*bb + *ba) / 2;
}

std::optional<Price> OrderBook::spread() const {
    auto bb = best_bid();
    auto ba = best_ask();
    if (!bb || !ba) return std::nullopt;
    return *ba - *bb;
}

Quantity OrderBook::bid_qty_at(Price p) const {
    auto it = bids_.find(p);
    return (it != bids_.end()) ? it->second.total_qty() : 0;
}

Quantity OrderBook::ask_qty_at(Price p) const {
    auto it = asks_.find(p);
    return (it != asks_.end()) ? it->second.total_qty() : 0;
}

std::vector<OrderBook::L2Entry> OrderBook::bid_levels(size_t n) const {
    std::vector<L2Entry> out;
    out.reserve(n);
    for (auto& [p, lvl] : bids_) {
        if (out.size() >= n) break;
        out.push_back({p, lvl.total_qty(), lvl.depth()});
    }
    return out;
}

std::vector<OrderBook::L2Entry> OrderBook::ask_levels(size_t n) const {
    std::vector<L2Entry> out;
    out.reserve(n);
    for (auto& [p, lvl] : asks_) {
        if (out.size() >= n) break;
        out.push_back({p, lvl.total_qty(), lvl.depth()});
    }
    return out;
}

// ─── Internal Helpers ─────────────────────────────────────────────────────────
template<typename BookSide>
std::vector<Trade> OrderBook::match_against(Order& aggressor, BookSide& passive_side) {
    std::vector<Trade> trades;

    while (!passive_side.empty() && aggressor.remaining() > 0) {
        auto& [best_price, best_level] = *passive_side.begin();

        // Check price crossing
        if (aggressor.type == OrderType::Limit) {
            if (aggressor.side == Side::Buy  && aggressor.price < best_price) break;
            if (aggressor.side == Side::Sell && aggressor.price > best_price) break;
        }

        // Match against front of level
        auto& passive = best_level.front();
        Quantity fill_qty = std::min(aggressor.remaining(), passive.remaining());

        Trade t;
        t.aggressor_id   = aggressor.id;
        t.passive_id     = passive.id;
        t.aggressor_side = aggressor.side;
        t.exec_price     = best_price;
        t.exec_qty       = fill_qty;
        t.timestamp      = aggressor.timestamp;

        aggressor.filled_qty += fill_qty;
        // Update passive order through level
        best_level.fill_front(fill_qty);
        if (best_level.empty()) {
            // Remove stale index entries before erasing level
            passive_side.erase(passive_side.begin());
        }

        trades.push_back(t);
        ++trade_count_;
    }
    return trades;
}

void OrderBook::insert_limit(Order& o) {
    PriceLevel::Iterator it;
    if (o.side == Side::Buy) {
        auto [level_it, inserted] = bids_.try_emplace(o.price, o.price);
        it = level_it->second.enqueue(o);
    } else {
        auto [level_it, inserted] = asks_.try_emplace(o.price, o.price);
        it = level_it->second.enqueue(o);
    }
    order_index_[o.id] = {o.side, o.price, it};
}

// Explicit template instantiations
template std::vector<Trade> OrderBook::match_against(Order&, BidMap&);
template std::vector<Trade> OrderBook::match_against(Order&, AskMap&);

} // namespace lob
