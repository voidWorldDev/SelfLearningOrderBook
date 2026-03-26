#include "include/price_level.hpp"

namespace lob {

PriceLevel::Iterator PriceLevel::enqueue(const Order& o) {
    total_qty_ += o.remaining();
    queue_.push_back(o);
    return std::prev(queue_.end());
}

void PriceLevel::remove(Iterator it) {
    total_qty_ -= it->remaining();
    queue_.erase(it);
}

void PriceLevel::fill_front(Quantity qty) {
    if (queue_.empty()) return;
    auto& front = queue_.front();
    Quantity actual = std::min(qty, front.remaining());
    front.filled_qty += actual;
    total_qty_       -= actual;
    if (front.is_filled()) {
        queue_.pop_front();
    }
}

} // namespace lob
