#pragma once
#include "order.hpp"
#include <list>
#include <unordered_map>

namespace lob {

// ─── PriceLevel ───────────────────────────────────────────────────────────────
// Represents all resting orders at a single price point.
// Maintains strict price-time priority via a FIFO queue.
class PriceLevel {
public:
    using Queue    = std::list<Order>;
    using Iterator = Queue::iterator;

    explicit PriceLevel(Price p) : price_(p), total_qty_(0) {}

    // Add an order to the back of the queue (time priority)
    Iterator enqueue(const Order& o);

    // Remove a specific order by iterator (O(1))
    void     remove(Iterator it);

    // Peek at the front order (best time priority)
    const Order& front() const { return queue_.front(); }
    Order&       front()       { return queue_.front(); }
    void         pop_front()   { total_qty_ -= queue_.front().remaining(); queue_.pop_front(); }

    // Partial fill the front order
    void fill_front(Quantity qty);

    Price    price()     const { return price_; }
    Quantity total_qty() const { return total_qty_; }
    bool     empty()     const { return queue_.empty(); }
    size_t   depth()     const { return queue_.size(); }

private:
    Price    price_;
    Quantity total_qty_;
    Queue    queue_;
};

} // namespace lob
