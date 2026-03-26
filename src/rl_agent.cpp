#include "include/rl_agent.hpp"
#include <algorithm>
#include <numeric>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <random>

namespace lob {

// ─── State Builder ────────────────────────────────────────────────────────────
StateVec build_state(const OrderBook&     book,
                     const InventoryState& inv,
                     const SpreadMetrics&  sm,
                     double time_remaining_s)
{
    StateVec s{};
    size_t idx = 0;

    // Book features: top NUM_LEVELS bid/ask price & qty (normalized)
    double mid = sm.mean_mid();
    if (mid < 1.0) mid = 1.0; // guard

    auto bid_lvls = book.bid_levels(NUM_LEVELS);
    auto ask_lvls = book.ask_levels(NUM_LEVELS);

    for (size_t i = 0; i < NUM_LEVELS; ++i) {
        if (i < bid_lvls.size()) {
            s[idx++] = static_cast<float>((mid - bid_lvls[i].price) / mid); // depth
            s[idx++] = static_cast<float>(std::log1p(bid_lvls[i].qty));
        } else {
            s[idx++] = 1.0f; s[idx++] = 0.0f;
        }
        if (i < ask_lvls.size()) {
            s[idx++] = static_cast<float>((ask_lvls[i].price - mid) / mid);
            s[idx++] = static_cast<float>(std::log1p(ask_lvls[i].qty));
        } else {
            s[idx++] = 1.0f; s[idx++] = 0.0f;
        }
    }

    // Inventory features
    s[idx++] = static_cast<float>(inv.position) / 100.0f;
    s[idx++] = static_cast<float>(inv.realized_pnl) / 1000.0f;
    s[idx++] = static_cast<float>(sm.mean_spread()) / static_cast<float>(mid);
    s[idx++] = static_cast<float>(sm.realized_vol());
    s[idx++] = static_cast<float>(std::log1p(inv.num_fills));
    s[idx++] = static_cast<float>(inv.num_buys)  / std::max(1.0f, static_cast<float>(inv.num_fills));
    s[idx++] = static_cast<float>(inv.num_sells) / std::max(1.0f, static_cast<float>(inv.num_fills));
    s[idx++] = static_cast<float>(std::max(0.0, time_remaining_s) / 60.0);

    return s;
}

// ─── Reward ───────────────────────────────────────────────────────────────────
float compute_reward(const InventoryState& before,
                     const InventoryState& after,
                     const std::vector<Trade>& fills,
                     Price mid_price,
                     const RewardConfig& cfg)
{
    float reward = 0.0f;

    // PnL delta
    double dpnl = (after.realized_pnl + after.unrealized_pnl)
                - (before.realized_pnl + before.unrealized_pnl);
    reward += static_cast<float>(cfg.pnl_weight * dpnl);

    // Inventory risk penalty (quadratic)
    double pos = static_cast<double>(after.position);
    reward -= static_cast<float>(cfg.inventory_penalty * pos * pos);

    // Spread capture reward per fill
    double mid_d = static_cast<double>(mid_price) / 100.0;
    for (auto& t : fills) {
        double exec_d = static_cast<double>(t.exec_price) / 100.0;
        double capture = (t.aggressor_side == Side::Sell)
                         ? (exec_d - mid_d) : (mid_d - exec_d);
        reward += static_cast<float>(cfg.spread_reward * capture);
        reward += static_cast<float>(cfg.fill_reward);
    }

    return reward;
}

// ─── NaiveMarketMaker ─────────────────────────────────────────────────────────
NaiveMarketMaker::NaiveMarketMaker(uint32_t id, int offset_ticks)
    : offset_ticks_(offset_ticks)
{
    agent_id_ = id;
}

ActionVec NaiveMarketMaker::act(const StateVec&) {
    return { static_cast<float>(offset_ticks_),
             static_cast<float>(offset_ticks_) };
}

// ─── SimpleQLearningAgent ─────────────────────────────────────────────────────
SimpleQLearningAgent::SimpleQLearningAgent(uint32_t id, float alpha, float gamma, float eps)
    : alpha_(alpha), gamma_(gamma), eps_(eps)
{
    agent_id_ = id;
    qtable_.assign(BINS * ACTIONS, 0.0f);
}

int SimpleQLearningAgent::discretize(const StateVec& s) const {
    // Simple hash: use inventory feature (index NUM_LEVELS*4) as primary bin
    float inv_norm = s[NUM_LEVELS * 4]; // position normalized to [-1,1]
    int bin = static_cast<int>((inv_norm + 1.0f) * 0.5f * (BINS - 1));
    return std::clamp(bin, 0, BINS - 1);
}

int SimpleQLearningAgent::greedy_action(int sb) const {
    int best = 0;
    float best_q = qtable_[sb * ACTIONS + 0];
    for (int a = 1; a < ACTIONS; ++a) {
        float q = qtable_[sb * ACTIONS + a];
        if (q > best_q) { best_q = q; best = a; }
    }
    return best;
}

ActionVec SimpleQLearningAgent::act(const StateVec& state) {
    last_state_ = state;
    int sb = discretize(state);

    int action;
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> udist(0.0f, 1.0f);
    if (udist(rng) < eps_) {
        std::uniform_int_distribution<int> adist(0, ACTIONS - 1);
        action = adist(rng);
    } else {
        action = greedy_action(sb);
    }

    // Decode action index to (bid_offset, ask_offset) in {-1, 0, 1}
    int bid_off = (action / 3) - 1; // -1, 0, 1
    int ask_off = (action % 3) - 1;
    last_action_ = { static_cast<float>(bid_off), static_cast<float>(ask_off) };
    has_prev_ = true;
    return last_action_;
}

void SimpleQLearningAgent::observe_reward(float reward, bool done) {
    if (!has_prev_) return;
    int sb   = discretize(last_state_);
    int ab   = static_cast<int>((last_action_[0] + 1) * 3 + (last_action_[1] + 1));
    ab = std::clamp(ab, 0, ACTIONS - 1);

    float target = reward;
    if (!done) {
        target += gamma_ * *std::max_element(
            qtable_.begin() + sb * ACTIONS,
            qtable_.begin() + sb * ACTIONS + ACTIONS);
    }
    float& q = qtable_[sb * ACTIONS + ab];
    q += alpha_ * (target - q);
    if (done) has_prev_ = false;
}

void SimpleQLearningAgent::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file: " + path);
    f.write(reinterpret_cast<const char*>(qtable_.data()),
            qtable_.size() * sizeof(float));
}

void SimpleQLearningAgent::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file: " + path);
    f.read(reinterpret_cast<char*>(qtable_.data()),
           qtable_.size() * sizeof(float));
}

void SimpleQLearningAgent::update_q(int sb, int ab, float r, int sb_next) {
    float best_next = *std::max_element(
        qtable_.begin() + sb_next * ACTIONS,
        qtable_.begin() + sb_next * ACTIONS + ACTIONS);
    float& q = qtable_[sb * ACTIONS + ab];
    q += alpha_ * (r + gamma_ * best_next - q);
}

} // namespace lob
