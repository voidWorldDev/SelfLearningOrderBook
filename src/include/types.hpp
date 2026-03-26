#pragma once
#include <cstdint>
#include <string>

namespace lob {

// ─── Order Side ───────────────────────────────────────────────────────────────
enum class Side : uint8_t {
    Buy  = 0,
    Sell = 1
};

inline Side opposite(Side s) {
    return (s == Side::Buy) ? Side::Sell : Side::Buy;
}

inline std::string to_string(Side s) {
    return (s == Side::Buy) ? "BUY" : "SELL";
}

// ─── Order Type ───────────────────────────────────────────────────────────────
enum class OrderType : uint8_t {
    Limit  = 0,
    Market = 1,
    Cancel = 2
};

inline std::string to_string(OrderType t) {
    switch (t) {
        case OrderType::Limit:  return "LIMIT";
        case OrderType::Market: return "MARKET";
        case OrderType::Cancel: return "CANCEL";
        default:                return "UNKNOWN";
    }
}

// ─── Order Status ─────────────────────────────────────────────────────────────
enum class OrderStatus : uint8_t {
    Pending   = 0,
    PartFill  = 1,
    Filled    = 2,
    Cancelled = 3,
    Rejected  = 4
};

// ─── Agent Action (RL) ────────────────────────────────────────────────────────
enum class AgentAction : uint8_t {
    PlaceBidLimit  = 0,
    PlaceAskLimit  = 1,
    PlaceMarketBuy = 2,
    PlaceMarketSell= 3,
    CancelBest     = 4,
    Hold           = 5
};

} // namespace lob
