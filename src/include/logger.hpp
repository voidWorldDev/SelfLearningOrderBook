#pragma once
#include "order_book.hpp"
#include "metrics.hpp"
#include <string>
#include <ostream>
#include <iostream>

namespace lob {

// Forward declare SimResult (defined in simulator.hpp)
struct SimResult;

// ─── Logger ───────────────────────────────────────────────────────────────────
class Logger {
public:
    enum class Level { DEBUG, INFO, WARN, ERROR };

    static Logger& get() { static Logger l; return l; }

    void set_level(Level l)           { level_ = l; }
    void set_stream(std::ostream& os) { out_ = &os; }

    template<typename... Args>
    void log(Level l, Args&&... args) {
        if (l < level_) return;
        *out_ << level_prefix(l);
        ((*out_ << std::forward<Args>(args)), ...);
        *out_ << '\n';
    }

    template<typename... Args> void info (Args&&... a) { log(Level::INFO,  std::forward<Args>(a)...); }
    template<typename... Args> void warn (Args&&... a) { log(Level::WARN,  std::forward<Args>(a)...); }
    template<typename... Args> void error(Args&&... a) { log(Level::ERROR, std::forward<Args>(a)...); }
    template<typename... Args> void debug(Args&&... a) { log(Level::DEBUG, std::forward<Args>(a)...); }

private:
    Level         level_ = Level::INFO;
    std::ostream* out_   = &std::cout;

    static const char* level_prefix(Level l) {
        switch (l) {
            case Level::DEBUG: return "[DBG] ";
            case Level::INFO:  return "[INF] ";
            case Level::WARN:  return "[WRN] ";
            case Level::ERROR: return "[ERR] ";
        }
        return "";
    }
};

// ─── Book Printer ─────────────────────────────────────────────────────────────
void print_book(const OrderBook& book, size_t levels = 5,
                std::ostream& os = std::cout);

void print_trade(const Trade& t, std::ostream& os = std::cout);

void print_sim_result(const SimResult& r, std::ostream& os = std::cout);

} // namespace lob
