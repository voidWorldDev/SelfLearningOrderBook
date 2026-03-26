#include "include/simulator.hpp"
#include "include/rl_agent.hpp"
#include "include/logger.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <chrono>

// ─────────────────────────────────────────────────────────────────────────────
// drut — Self-Learning Limit Order Book Simulator
//
// Entry point: configures the simulation, attaches RL agents, runs the loop,
// and prints a structured result summary.
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    using namespace lob;

    // ── Banner ────────────────────────────────────────────────────────────────
    std::cout
        << "\n"
        << "  ██████╗ ██████╗ ██╗   ██╗████████╗\n"
        << "  ██╔══██╗██╔══██╗██║   ██║╚══██╔══╝\n"
        << "  ██║  ██║██████╔╝██║   ██║   ██║   \n"
        << "  ██║  ██║██╔══██╗██║   ██║   ██║   \n"
        << "  ██████╔╝██║  ██║╚██████╔╝   ██║   \n"
        << "  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝    ╚═╝   \n"
        << "  Self-Learning Limit Order Book Simulator\n"
        << "  ─────────────────────────────────────────\n\n";

    // ── Parse simple CLI args ─────────────────────────────────────────────────
    uint64_t duration_s    = 60;
    bool     show_book     = false;
    std::string agent_type = "qlearn"; // "naive" | "qlearn"

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--duration" && i + 1 < argc) {
            duration_s = std::stoull(argv[++i]);
        } else if (arg == "--show-book") {
            show_book = true;
        } else if (arg == "--agent" && i + 1 < argc) {
            agent_type = argv[++i];
        } else if (arg == "--help") {
            std::cout
                << "Usage: drut [OPTIONS]\n"
                << "  --duration <secs>   Simulation duration in seconds (default: 60)\n"
                << "  --agent <type>      Agent type: naive | qlearn (default: qlearn)\n"
                << "  --show-book         Print L2 book snapshots every second\n"
                << "  --help              Show this message\n\n";
            return 0;
        }
    }

    // ── Configure simulation ──────────────────────────────────────────────────
    SimConfig cfg;
    cfg.duration_ns           = duration_s * 1'000'000'000ULL;
    cfg.tick_dt_ns            = 1'000'000ULL; // 1 ms ticks
    cfg.log_trades            = true;
    cfg.log_book_snapshot     = show_book;
    cfg.snapshot_interval_ns  = 1'000'000'000ULL;

    // Stochastic flow: moderate activity
    cfg.flow_cfg.lambda_limit_buy   = 12.0;
    cfg.flow_cfg.lambda_limit_sell  = 12.0;
    cfg.flow_cfg.lambda_market_buy  =  4.0;
    cfg.flow_cfg.lambda_market_sell =  4.0;
    cfg.flow_cfg.lambda_cancel      =  6.0;
    cfg.flow_cfg.init_mid_price     = 10000.0; // $100.00
    cfg.flow_cfg.tick_size          = 1.0;

    // Reward shaping
    cfg.reward_cfg.pnl_weight        = 1.0;
    cfg.reward_cfg.inventory_penalty = 0.01;
    cfg.reward_cfg.spread_reward     = 0.5;
    cfg.reward_cfg.fill_reward       = 0.05;

    // ── Create RL agent ───────────────────────────────────────────────────────
    std::shared_ptr<RLAgent> agent;
    if (agent_type == "naive") {
        agent = std::make_shared<NaiveMarketMaker>(1, /*offset_ticks=*/1);
        std::cout << "  Agent: NaiveMarketMaker (always quotes +/-1 tick)\n\n";
    } else {
        agent = std::make_shared<SimpleQLearningAgent>(
            1, /*alpha=*/0.01f, /*gamma=*/0.99f, /*eps=*/0.10f);
        std::cout << "  Agent: Q-Learning Market Maker (alpha=0.01, gamma=0.99, eps=0.10)\n\n";
    }

    // ── Run ───────────────────────────────────────────────────────────────────
    Simulator sim(cfg);
    sim.add_agent(agent);

    std::cout << "  Starting " << duration_s << "s simulation...\n\n";
    auto wall_start = std::chrono::high_resolution_clock::now();

    SimResult result = sim.run();

    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_s = std::chrono::duration<double>(wall_end - wall_start).count();

    // ── Results ───────────────────────────────────────────────────────────────
    print_sim_result(result, std::cout);

    std::cout << "  Wall-clock runtime : " << wall_s << "s\n";
    std::cout << "  Simulated/Real     : " << static_cast<double>(duration_s) / wall_s
              << "x\n\n";

    // Final L2 book snapshot
    std::cout << "  Final Order Book State:\n";
    print_book(sim.book(), 5, std::cout);

    return 0;
}
