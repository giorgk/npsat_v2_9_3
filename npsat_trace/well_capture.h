#ifndef NPSAT_TRACE_WELL_CAPTURE_H
#define NPSAT_TRACE_WELL_CAPTURE_H

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace npsat_trace {

// Purely local model: coordinates in [0,1]^3, rates in physical volume/time.
// It owns no mesh, MPI communicator, or particle state.
struct CapturePoint {
    std::array<double, 3> xi;
    double z;
};
struct CaptureWell {
    unsigned int id;
    double u, v, bottom, top, rate, lambda;
};
struct CapturePath {
    double weight;
    bool exited = false;
    std::vector<CapturePoint> points;
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
    double inflow = 0.0;
    double relative_budget_error = 0.0;
    double relative_capture_error = 0.0;
    double unresolved_flow_fraction = 0.0;
    bool calibrated = false;

    int at(const CapturePoint &p) const {
        double best = face_score(p);
        int selected = -1;
        for (unsigned int j = 0; j < wells.size(); ++j) {
            const auto &w = wells[j];
            if (p.z < w.bottom || p.z > w.top) continue;
            const double dx = p.xi[0] - w.u, dy = p.xi[1] - w.v;
            const double score = w.lambda * w.rate /
                (dx*dx + dy*dy + regularization*regularization);
            if (score > best) { best = score; selected = static_cast<int>(j); }
        }
        return selected;
    }

    // Search left-to-right along a straight reference-space chord. Bounds on
    // well and face scores reject intervals that cannot contain capture. This
    // detects a region between endpoints even when both endpoints miss it.
    CaptureHit intersect(const CapturePoint &a, const CapturePoint &b) const {
        return search(a, b, 0.0, 1.0, 0);
    }

    // Joint coordinate fitting: each path is assigned only to its first well.
    // Deterministic quadrature has a finite flux quantum; fitting below that
    // resolution is neither meaningful nor worth additional computation.
    void calibrate(const std::vector<CapturePath> &paths) {
        calibrated = false;
        double sink = 0.0, face_out = 0.0, quantum = 0.0;
        for (const auto &w : wells) sink += w.rate;
        for (double q : outward) face_out += q;
        for (const auto &p : paths) quantum = std::max(quantum, p.weight);
        const double scale = std::max(inflow, sink + face_out);
        relative_budget_error = scale > 0 ? std::abs(inflow-sink-face_out)/scale : 0;
        if (wells.empty()) return;
        // No fabricated storage/recharge term: inconsistent available budgets
        // retain lambda=1 and must be diagnosed by the caller.
        if (inflow <= 1.e-12 || relative_budget_error > 0.01 || paths.empty()) return;
        auto evaluate = [&]() {
            std::vector<double> captured(wells.size(), 0.0);
            double unresolved = 0.0;
            for (const auto &path : paths) {
                if (path.points.empty()) { unresolved += path.weight; continue; }
                int selected = at(path.points.front());
                for (unsigned int k = 1; selected < 0 && k < path.points.size(); ++k)
                    selected = intersect(path.points[k-1], path.points[k]).well;
                if (selected >= 0) captured[selected] += path.weight;
                else if (!path.exited) unresolved += path.weight;
            }
            unresolved_flow_fraction = unresolved / inflow;
            return captured;
        };
        auto error = [&](const std::vector<double> &captured) {
            double result = 0.0;
            for (unsigned int j = 0; j < wells.size(); ++j)
                result = std::max(result, std::abs(captured[j] - wells[j].rate));
            return result;
        };
        auto objective = [&](const std::vector<double> &captured) {
            double result = 0.0;
            for (unsigned int j = 0; j < wells.size(); ++j) {
                const double residual=(captured[j]-wells[j].rate)/inflow;
                result += residual*residual;
            }
            return result;
        };
        std::vector<double> best(wells.size(), 1.0);
        const auto initial=evaluate();
        double best_error = error(initial), best_objective=objective(initial);
        const double tolerance = std::max(0.01 * inflow, quantum);
        for (unsigned int sweep = 0; sweep < 3 && best_error > tolerance; ++sweep) {
            for (unsigned int j = 0; j < wells.size(); ++j) {
                double lo = -24.0, hi = 24.0;
                for (unsigned int iteration = 0; iteration < 14; ++iteration) {
                    const double mid = 0.5 * (lo + hi);
                    wells[j].lambda = std::exp(mid);
                    const auto captured = evaluate();
                    const double current_error = error(captured);
                    const double current_objective=objective(captured);
                    if (current_objective < best_objective) {
                        best_error = current_error;
                        best_objective=current_objective;
                        for (unsigned int k = 0; k < wells.size(); ++k) best[k] = wells[k].lambda;
                    }
                    if (captured[j] < wells[j].rate) lo = mid;
                    else hi = mid;
                }
                for (unsigned int k = 0; k < wells.size(); ++k) wells[k].lambda = best[k];
            }
            for (unsigned int j = 0; j < wells.size(); ++j) wells[j].lambda = best[j];
        }
        for (unsigned int j = 0; j < wells.size(); ++j) wells[j].lambda = best[j];
        relative_capture_error = best_error / inflow;
        evaluate();
        calibrated = best_error <= tolerance && unresolved_flow_fraction*inflow <= tolerance;
    }

private:
    static CapturePoint interpolate(const CapturePoint &a, const CapturePoint &b, double t) {
        CapturePoint p;
        for (unsigned int d = 0; d < 3; ++d) p.xi[d] = a.xi[d] + t*(b.xi[d]-a.xi[d]);
        p.z = a.z + t*(b.z-a.z);
        return p;
    }
    double face_score(const CapturePoint &p) const {
        double result = 0.0;
        for (unsigned int f = 0; f < 6; ++f) {
            const double d = (f%2) ? 1.0-p.xi[f/2] : p.xi[f/2];
            result += outward[f] / (d*d + regularization*regularization);
        }
        return result;
    }
    bool possible(const CapturePoint &a, const CapturePoint &b) const {
        double face_lower = 0.0;
        for (unsigned int f = 0; f < 6; ++f) {
            const double da = (f%2) ? 1.0-a.xi[f/2] : a.xi[f/2];
            const double db = (f%2) ? 1.0-b.xi[f/2] : b.xi[f/2];
            const double distance = std::max(std::abs(da), std::abs(db));
            face_lower += outward[f] / (distance*distance + regularization*regularization);
        }
        for (const auto &w : wells) {
            if (std::max(a.z,b.z) < w.bottom || std::min(a.z,b.z) > w.top) continue;
            const double dx = std::max(0.0, std::max(std::min(a.xi[0],b.xi[0])-w.u,
                                                    w.u-std::max(a.xi[0],b.xi[0])));
            const double dy = std::max(0.0, std::max(std::min(a.xi[1],b.xi[1])-w.v,
                                                    w.v-std::max(a.xi[1],b.xi[1])));
            if (w.lambda*w.rate/(dx*dx+dy*dy+regularization*regularization) > face_lower)
                return true;
        }
        return false;
    }
    CaptureHit search(const CapturePoint &a, const CapturePoint &b,
                      double begin, double end, unsigned int depth) const {
        CaptureHit hit;
        hit.well = at(a);
        if (hit.well >= 0) { hit.fraction = begin; return hit; }
        if (!possible(a,b)) return hit;
        double span = 0;
        for (unsigned int d = 0; d < 3; ++d) span = std::max(span, std::abs(a.xi[d]-b.xi[d]));
        if (depth >= 24 || span <= 1.e-7) {
            // At numerical resolution, also inspect the closest horizontal
            // approach to each well and the endpoints of its screen interval.
            double first = 2.0;
            auto inspect = [&](double t) {
                if (t < 0 || t > 1) return;
                const int j = at(interpolate(a,b,t));
                if (j >= 0 && t < first) { first = t; hit.well = j; }
            };
            inspect(0.5); inspect(1.0);
            for (const auto &w : wells) {
                const double dx=b.xi[0]-a.xi[0], dy=b.xi[1]-a.xi[1];
                const double dd=dx*dx+dy*dy;
                if (dd > 0) inspect(std::max(0.0,std::min(1.0,
                    ((w.u-a.xi[0])*dx+(w.v-a.xi[1])*dy)/dd)));
                if (b.z != a.z) { inspect((w.bottom-a.z)/(b.z-a.z)); inspect((w.top-a.z)/(b.z-a.z)); }
            }
            if (hit.well >= 0) hit.fraction = begin + first*(end-begin);
            return hit;
        }
        const auto middle = interpolate(a,b,0.5);
        const double half = 0.5*(begin+end);
        hit = search(a,middle,begin,half,depth+1);
        return hit.well >= 0 ? hit : search(middle,b,half,end,depth+1);
    }
};
}
#endif
