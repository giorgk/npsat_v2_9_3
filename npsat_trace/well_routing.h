#ifndef NPSAT_TRACE_WELL_ROUTING_H
#define NPSAT_TRACE_WELL_ROUTING_H

#include <algorithm>
#include <cmath>

namespace npsat_trace {
// Signed axial flows are upward-positive; exchange is positive into aquifer.
// All flows must already be adjusted for tracking direction. A uniform sample
// selects an outlet under the perfectly mixed well-segment approximation.
enum class WellRoute { up, down, aquifer, stagnant, invalid };

inline WellRoute select_well_route(double exchange, double bottom, double top,
                                  double sample) {
    if (!std::isfinite(exchange) || !std::isfinite(bottom) ||
        !std::isfinite(top) || !(sample >= 0.0 && sample < 1.0))
        return WellRoute::invalid;
    const double scale = std::max(std::abs(exchange),
                                  std::max(std::abs(bottom), std::abs(top)));
    const double tolerance = 1.e-12 + 1.e-10 * scale;
    if (std::abs(bottom - top - exchange) > tolerance)
        return WellRoute::invalid;
    if (scale <= 1.e-12)
        return WellRoute::stagnant;
    // Normalize before summation to avoid overflow for large flow rates.
    const double up = std::max(top / scale, 0.0);
    const double down = std::max(-bottom / scale, 0.0);
    const double aquifer = std::max(exchange / scale, 0.0);
    const double total = up + down + aquifer;
    if (total <= 0.0)
        return WellRoute::invalid;
    const double choice = sample * total;
    if (choice < up) return WellRoute::up;
    if (choice < up + down) return WellRoute::down;
    return WellRoute::aquifer;
}
}
#endif
