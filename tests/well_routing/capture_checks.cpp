// Small C++11-only numerical check; no deal.II, MPI, or simulation files.
// g++ -std=c++11 -O2 -I. tests/well_routing/capture_checks.cpp -o /tmp/capture_checks
#include "npsat_trace/well_capture.h"
#include <iostream>
#include <stdexcept>

void check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }

int main() {
    using namespace npsat_trace;
    WellCaptureModel model;
    model.inflow=100;
    model.outward[1]=80;
    model.wells.push_back({7,.5,.5,0,1,20,1});
    std::vector<CapturePath> paths;
    for (unsigned int i=0;i<100;++i) {
        CapturePath path;
        path.weight=1;
        path.exited=true;
        const double y=(i+.5)/100;
        path.points.push_back({{{0,y,.5}},.5});
        path.points.push_back({{{1,y,.5}},.5});
        paths.push_back(path);
    }
    model.calibrate(paths);
    check(model.calibrated && model.relative_capture_error<=.01,"single-well flux calibration");
    const CapturePoint a{{{0,.5,.5}},.5}, b{{{1,.5,.5}},.5};
    check(model.at(a)<0 && model.at(b)<0,"endpoints should miss capture region");
    const auto hit=model.intersect(a,b);
    check(hit.well==0 && hit.fraction>0 && hit.fraction<1,"must detect interior capture");
    const CapturePoint above{{{.5,.5,.5}},2};
    check(model.at(above)<0,"screen elevation must gate capture");
    model.wells.clear();
    model.wells.push_back({7,.5,.25,0,1,10,1});
    model.wells.push_back({8,.5,.75,0,1,10,1});
    model.calibrate(paths);
    check(model.calibrated && model.relative_capture_error<=.01,"joint multi-well calibration");
    WellCaptureModel incomplete;
    incomplete.inflow=10;
    incomplete.outward[1]=80;
    incomplete.wells.push_back({7,.5,.5,0,1,20,1});
    incomplete.calibrate(paths);
    check(!incomplete.calibrated && incomplete.wells[0].lambda==1,"unbalanced budget fallback");
    std::cout << "capture checks passed: calibration, joint allocation, interior entry, screen, budget fallback\n";
}
