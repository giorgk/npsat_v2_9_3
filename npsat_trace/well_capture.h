#ifndef NPSAT_TRACE_WELL_CAPTURE_H
#define NPSAT_TRACE_WELL_CAPTURE_H

#include <algorithm>
#include <array>
#include <vector>

namespace npsat_trace {

// Reference coordinates are in [0,1]^3. Elevations and rates are physical.
struct CapturePoint {
    std::array<double, 3> xi;
    double z;
};

inline bool capture_inside(const CapturePoint &p) {
    for (double v : p.xi)
        if (v < 0.0 || v > 1.0) return false;
    return true;
}

struct CaptureWell {
    unsigned int id;
    CapturePoint bottom, top;
    double rate;
};

struct CaptureHit {
    int well = -1; // index in wells, not global well id
    double fraction = 1.0;
};

class WellCaptureModel {
public:
    static constexpr double regularization = 1.e-6;
    std::array<double, 6> outward{{0,0,0,0,0,0}};
    std::vector<CaptureWell> wells;

    int at(const CapturePoint &p) const {
        if (!capture_inside(p)) return -1;
        double best = face_score(p);
        int selected = -1;
        for (unsigned int j = 0; j < wells.size(); ++j) {
            const auto &w = wells[j];
            const double screen_length = w.top.z - w.bottom.z;
            const double height_fraction = screen_length > 0.0
                ? (std::max(w.bottom.z, std::min(p.z, w.top.z)) - w.bottom.z) / screen_length
                : 0.0;
            double distance_squared = 0.0;
            for (unsigned int d = 0; d < 3; ++d) {
                const double nearest = w.bottom.xi[d] + height_fraction * (w.top.xi[d] - w.bottom.xi[d]);
                const double distance = p.xi[d] - nearest;
                distance_squared += distance * distance;
            }
            const double score = w.rate / (distance_squared + regularization * regularization);
            if (score > best) {
                best = score;
                selected = static_cast<int>(j);
            }
        }
        return selected;
    }

private:
    double face_score(const CapturePoint &p) const {
        double result = 0.0;
        for (unsigned int f = 0; f < 6; ++f) {
            const double d = (f % 2) ? 1.0 - p.xi[f / 2] : p.xi[f / 2];
            result += outward[f] / (d * d + regularization * regularization);
        }
        return result;
    }
};

}
#endif
