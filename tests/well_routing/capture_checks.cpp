// Small C++11-only numerical check; no deal.II, MPI, or simulation files.
#include "npsat_trace/well_capture.h"
#include <iostream>
#include <stdexcept>

void check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }

int main() {
    using namespace npsat_trace;
    WellCaptureModel model;
    model.outward[1] = 80;
    model.wells.push_back({7, {{{.5, .5, .4}}, 4}, {{{.5, .5, .6}}, 6}, 20});
    check(model.at({{{.5, .5, .5}}, 5}) == 0, "screen interior should capture");
    check(model.at({{{.5, .5, .7}}, 7}) == 0, "top endpoint should attract");
    check(model.at({{{.5, .5, .3}}, 3}) == 0, "bottom endpoint should attract");
    check(model.at({{{.5, .5, .9}}, 9}) < 0, "distant top endpoint should lose to outflow");
    check(model.at({{{.5, .5, .1}}, 1}) < 0, "distant bottom endpoint should lose to outflow");
    model.outward[1] = 0;
    check(model.at({{{.5, .5, 1.01}}, 10.1}) < 0, "outside sample should not capture");
    std::cout << "capture checks passed\n";
}
