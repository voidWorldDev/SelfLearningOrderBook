#include "include/metrics.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>

namespace lob {

// ─── InventoryTracker ─────────────────────────────────────────────────────────
InventoryTracker::InventoryTracker(int64_t max_position, double fee_rate)
    : max_pos_(max_position), fee_rate_(fee_rate) {}

void InventoryTracker::on_fill(const Trade& t, Side aggressor_side) {
    double price = static_cast<double>(t.exec_price) / 100.0;
    double qty   = static_cast<double>(t.exec_qty);
    double notional = price * qty;
    double fee = notional * fee_rate_;

    // Determine if we were buyer or seller
    bool is_buy = (aggressor_side == Side::Buy);

    if (is_buy) {
        double old_pos = static_cast<double>(state_.position);
        double new_pos = old_pos + qty;
        // Update average cost
        if (new_pos > 0)
            avg_cost_ = (avg_cost_ * old_pos + notional) / new_pos;
        state_.position += static_cast<int64_t>(qty);
        ++state_.num_buys;
    } else {
        // Selling: realize PnL on the portion of position we're closing
        double close_qty = std::min(qty, static_cast<double>(std::max(state_.position, int64_t(0))));
        if (close_qty > 0) {
            state_.realized_pnl += close_qty * (price - avg_cost_);
        }
        state_.position -= static_cast<int64_t>(qty);
        ++state_.num_sells;
    }

    state_.total_fees += fee;
    state_.realized_pnl -= fee;
    ++state_.num_fills;
}

void InventoryTracker::update_mid(Price mid) {
    last_mid_ = mid;
    double price = static_cast<double>(mid) / 100.0;
    state_.unrealized_pnl = static_cast<double>(state_.position) * (price - avg_cost_);
}

bool InventoryTracker::at_limit() const {
    return std::abs(state_.position) >= max_pos_;
}

double InventoryTracker::inventory_risk() const {
    return std::abs(static_cast<double>(state_.position))
           * (static_cast<double>(last_mid_) / 100.0);
}

void InventoryTracker::reset() {
    state_    = {};
    avg_cost_ = 0.0;
    last_mid_ = 0;
}

// ─── SpreadMetrics ────────────────────────────────────────────────────────────
SpreadMetrics::SpreadMetrics(size_t window) : window_(window) {}

void SpreadMetrics::record(const SpreadSnapshot& snap) {
    if (!history_.empty()) {
        double prev_mid = static_cast<double>(history_.back().mid);
        double curr_mid = static_cast<double>(snap.mid);
        if (prev_mid > 0) {
            mid_returns_.push_back((curr_mid - prev_mid) / prev_mid);
            if (mid_returns_.size() > window_) mid_returns_.pop_front();
        }
    }
    history_.push_back(snap);
    if (history_.size() > window_) history_.pop_front();
}

double SpreadMetrics::mean_spread() const {
    if (history_.empty()) return 0.0;
    double sum = 0;
    for (auto& s : history_) sum += static_cast<double>(s.spread);
    return sum / static_cast<double>(history_.size());
}

double SpreadMetrics::spread_vol() const {
    if (history_.size() < 2) return 0.0;
    double mu = mean_spread();
    double var = 0;
    for (auto& s : history_) {
        double d = static_cast<double>(s.spread) - mu;
        var += d * d;
    }
    return std::sqrt(var / static_cast<double>(history_.size()));
}

double SpreadMetrics::mean_mid() const {
    if (history_.empty()) return 0.0;
    double sum = 0;
    for (auto& s : history_) sum += static_cast<double>(s.mid);
    return sum / static_cast<double>(history_.size());
}

double SpreadMetrics::realized_vol() const {
    if (mid_returns_.size() < 2) return 0.0;
    double sum = 0, sum2 = 0;
    for (double r : mid_returns_) { sum += r; sum2 += r * r; }
    double n  = static_cast<double>(mid_returns_.size());
    double var = sum2/n - (sum/n)*(sum/n);
    return std::sqrt(std::max(var, 0.0));
}

// ─── FillProbabilityEstimator ─────────────────────────────────────────────────
FillProbabilityEstimator::FillProbabilityEstimator(int max_ticks)
    : max_ticks_(max_ticks) {}

void FillProbabilityEstimator::record_placement(
    OrderId id, int ticks, Timestamp placed_at)
{
    open_[id] = {std::min(ticks, max_ticks_), placed_at, false, 0};
}

void FillProbabilityEstimator::record_fill(OrderId id, Timestamp filled_at) {
    auto it = open_.find(id);
    if (it == open_.end()) return;
    it->second.filled = true;
    it->second.fill_latency_ns = filled_at - it->second.placed_at;
    closed_.push_back(it->second);
    open_.erase(it);
}

void FillProbabilityEstimator::record_cancel(OrderId id) {
    auto it = open_.find(id);
    if (it == open_.end()) return;
    closed_.push_back(it->second);
    open_.erase(it);
}

double FillProbabilityEstimator::fill_prob(int ticks) const {
    int total = 0, filled = 0;
    for (auto& p : closed_) {
        if (p.ticks == ticks) {
            ++total;
            if (p.filled) ++filled;
        }
    }
    return total > 0 ? static_cast<double>(filled) / total : 0.0;
}

double FillProbabilityEstimator::avg_fill_time_ms(int ticks) const {
    double sum = 0; int n = 0;
    for (auto& p : closed_) {
        if (p.ticks == ticks && p.filled) {
            sum += static_cast<double>(p.fill_latency_ns) * 1e-6;
            ++n;
        }
    }
    return n > 0 ? sum / n : 0.0;
}

// ─── AlphaDecayAnalyzer ───────────────────────────────────────────────────────
double AlphaDecayAnalyzer::mean_shortfall() const {
    if (obs_.empty()) return 0.0;
    double sum = 0;
    for (auto& o : obs_) {
        // Implementation shortfall: exec vs mid at placement
        double sign = (o.side == Side::Buy) ? 1.0 : -1.0;
        sum += sign * static_cast<double>(o.exec_price - o.mid_at_placement);
    }
    return sum / static_cast<double>(obs_.size());
}

double AlphaDecayAnalyzer::mean_spread_capture() const {
    if (obs_.empty()) return 0.0;
    double sum = 0;
    for (auto& o : obs_) {
        double mid = static_cast<double>(o.mid_at_fill);
        double exec = static_cast<double>(o.exec_price);
        // Sell above mid = positive capture; buy below mid = positive capture
        double capture = (o.side == Side::Sell) ? (exec - mid) : (mid - exec);
        sum += capture;
    }
    return sum / static_cast<double>(obs_.size());
}

} // namespace lob
