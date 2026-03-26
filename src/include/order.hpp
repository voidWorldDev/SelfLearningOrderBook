#pragma once
#include "types.hpp"
#include <cstdint>
#include <chrono>

namespace lob {

using OrderId    = uint64_t;
using Price      = int64_t;   // fixed-point: price * 100 (e.g. 10050 = $100.50)
using Quantity   = uint64_t;
using Timestamp  = uint64_t;  // nanoseconds since epoch

constexpr Price   PRICE_TICK  = 1;      // 1 cent
constexpr Price   INVALID_PRICE = -1;

// ─── Core Order ───────────────────────────────────────────────────────────────
struct Order {
    OrderId    id;
    Side       side;
    OrderType  type;
    OrderStatus status;

    Price      price;      // only valid for Limit orders
    Quantity   qty;        // original quantity
    Quantity   filled_qty; // cumulative filled
    Timestamp  timestamp;  // nanosecond arrival time

    uint32_t   agent_id;   // 0 = stochastic flow; >0 = RL agent

    Order() = default;
    Order(OrderId id_, Side side_, OrderType type_,
          Price price_, Quantity qty_, Timestamp ts, uint32_t agent = 0)
        : id(id_), side(side_), type(type_), status(OrderStatus::Pending),
          price(price_), qty(qty_), filled_qty(0),
          timestamp(ts), agent_id(agent) {}

    Quantity remaining() const { return qty - filled_qty; }
    bool     is_filled() const { return filled_qty >= qty; }
};

// ─── Trade / Fill Report ──────────────────────────────────────────────────────
struct Trade {
    OrderId   aggressor_id;
    OrderId   passive_id;
    Side      aggressor_side;
    Price     exec_price;
    Quantity  exec_qty;
    Timestamp timestamp;
};

} // namespace lob
