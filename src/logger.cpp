#include "include/logger.hpp"
#include "include/simulator.hpp"
#include <iomanip>
#include <sstream>

namespace lob {

// ─── Book Printer ─────────────────────────────────────────────────────────────
void print_book(const OrderBook& book, size_t levels, std::ostream& os) {
    auto asks = book.ask_levels(levels);
    auto bids = book.bid_levels(levels);

    os << std::fixed << std::setprecision(2);
    os << "\n";
    os << "  ┌─────────────────────────────────────────────┐\n";
    os << "  │           LIMIT ORDER BOOK (L2)             │\n";
    os << "  ├──────────┬──────────────┬────────────────────┤\n";
    os << "  │  SIDE    │  PRICE ($)   │  QUANTITY          │\n";
    os << "  ├──────────┼──────────────┼────────────────────┤\n";

    // Print asks in reverse (highest first in display, lowest at spread)
    for (int i = static_cast<int>(asks.size()) - 1; i >= 0; --i) {
        auto& a = asks[i];
        os << "  │  \033[31mASK\033[0m      │  "
           << std::setw(10) << static_cast<double>(a.price) / 100.0
           << "  │  " << std::setw(12) << a.qty
           << " (" << a.num_orders << " orders)   │\n";
    }

    // Spread indicator
    if (book.best_bid() && book.best_ask()) {
        double spread_cents = static_cast<double>(*book.spread());
        double mid          = static_cast<double>(*book.mid_price()) / 100.0;
        os << "  ├──────────┴──────────────┴────────────────────┤\n";
        os << "  │  mid=" << std::setw(8) << mid
           << "  spread=" << std::setw(4) << spread_cents << " ticks"
           << "              │\n";
        os << "  ├──────────┬──────────────┬────────────────────┤\n";
    }

    // Print bids
    for (auto& b : bids) {
        os << "  │  \033[32mBID\033[0m      │  "
           << std::setw(10) << static_cast<double>(b.price) / 100.0
           << "  │  " << std::setw(12) << b.qty
           << " (" << b.num_orders << " orders)   │\n";
    }

    os << "  └──────────┴──────────────┴────────────────────┘\n";
    os << "  Orders: " << book.order_count()
       << "  Trades: " << book.trade_count()
       << "  Cancels: " << book.cancel_count() << "\n\n";
}

// ─── Trade Printer ────────────────────────────────────────────────────────────
void print_trade(const Trade& t, std::ostream& os) {
    os << std::fixed << std::setprecision(2);
    os << "[TRADE] "
       << to_string(t.aggressor_side)
       << " agg=" << t.aggressor_id
       << " pas=" << t.passive_id
       << " price=$" << static_cast<double>(t.exec_price) / 100.0
       << " qty=" << t.exec_qty
       << " ts=" << t.timestamp << "ns\n";
}

// ─── Sim Result Printer ───────────────────────────────────────────────────────
void print_sim_result(const SimResult& r, std::ostream& os) {
    os << std::fixed << std::setprecision(4);
    os << "\n╔══════════════════════════════════════════════╗\n";
    os << "║         SIMULATION RESULTS SUMMARY           ║\n";
    os << "╠══════════════════════════════════════════════╣\n";
    os << "║  Total Orders    : " << std::setw(10) << r.total_orders  << "              ║\n";
    os << "║  Total Trades    : " << std::setw(10) << r.total_trades  << "              ║\n";
    os << "║  Realized Vol    : " << std::setw(10) << r.realized_vol  << "              ║\n";
    os << "║  Mean Spread(tk) : " << std::setw(10) << r.mean_spread_ticks << "          ║\n";
    os << "║  Fill Prob @1tk  : " << std::setw(10) << r.mean_fill_prob    << "          ║\n";
    os << "╠══════════════════════════════════════════════╣\n";
    os << "║         AGENT INVENTORY & PnL                ║\n";
    os << "╠══════════════════════════════════════════════╣\n";
    os << "║  Net Position    : " << std::setw(10) << r.agent_inventory.position    << "              ║\n";
    os << "║  Realized PnL    : " << std::setw(10) << r.agent_inventory.realized_pnl << "          ║\n";
    os << "║  Unrealized PnL  : " << std::setw(10) << r.agent_inventory.unrealized_pnl << "        ║\n";
    os << "║  Total Fees      : " << std::setw(10) << r.agent_inventory.total_fees  << "          ║\n";
    os << "║  Total PnL       : " << std::setw(10) << r.agent_pnl                  << "          ║\n";
    os << "║  Fills           : " << std::setw(10) << r.agent_inventory.num_fills  << "              ║\n";
    os << "║  Buys            : " << std::setw(10) << r.agent_inventory.num_buys   << "              ║\n";
    os << "║  Sells           : " << std::setw(10) << r.agent_inventory.num_sells  << "              ║\n";
    os << "╚══════════════════════════════════════════════╝\n\n";
}

} // namespace lob
