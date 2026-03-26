#pragma once
#include "price_level.hpp"
#include <map>
#include <unordered_map>
#include <vector>
#include <optional>
#include <functional>

namespace lob {

// ─── OrderBook ────────────────────────────────────────────────────────────────
// Level-2 limit order book with price-time priority matching.
//
// Bids: sorted descending by price (highest bid at front)
// Asks: sorted ascending  by price (lowest ask at front)
class OrderBook {
public:
    using BidMap = std::map<Price, PriceLevel, std::greater<Price>>;
    using AskMap = std::map<Price, PriceLevel, std::less<Price>>;

    using TradeCallback  = std::function<void(const Trade&)>;
    using CancelCallback = std::function<void(OrderId)>;

    OrderBook() = default;

    // ── Core Operations ───────────────────────────────────────────────────────
    std::vector<Trade> add_limit_order(Order order);
    std::vector<Trade> add_market_order(Order order);
    bool               cancel_order(OrderId id);

    // ── Book State ────────────────────────────────────────────────────────────
    std::optional<Price> best_bid() const;
    std::optional<Price> best_ask() const;
    std::optional<Price> mid_price() const;
    std::optional<Price> spread()    const;

    Quantity bid_qty_at(Price p) const;
    Quantity ask_qty_at(Price p) const;

    // Level-2 snapshot: top N price levels each side
    struct L2Entry { Price price; Quantity qty; size_t num_orders; };
    std::vector<L2Entry> bid_levels(size_t n = 5) const;
    std::vector<L2Entry> ask_levels(size_t n = 5) const;

    // ── Stats ─────────────────────────────────────────────────────────────────
    uint64_t order_count()  const { return order_count_; }
    uint64_t trade_count()  const { return trade_count_; }
    uint64_t cancel_count() const { return cancel_count_; }

    const BidMap& bids() const { return bids_; }
    const AskMap& asks() const { return asks_; }

private:
    BidMap bids_;
    AskMap asks_;

    struct OrderMeta {
        Side           side;
        Price          price;
        PriceLevel::Iterator it;
    };
    std::unordered_map<OrderId, OrderMeta> order_index_;

    uint64_t order_count_  = 0;
    uint64_t trade_count_  = 0;
    uint64_t cancel_count_ = 0;

    // Internal matching
    template<typename BookSide>
    std::vector<Trade> match_against(Order& aggressor, BookSide& passive_side);

    void insert_limit(Order& o);
};

} // namespace lob
