#ifndef NPSAT_TRACE_CAPTURE_IMPL_H
#define NPSAT_TRACE_CAPTURE_IMPL_H

namespace npsat_trace {
template<int dim>
CapturePoint capture_point(const CellVelocityCacheRT0Split3D<dim> &cache,
                           const dealii::Point<dim> &x) {
    dealii::Point<dim> r;
    AssertThrow(cache.capture_reference_point(x,r),
                dealii::ExcMessage("Well capture: inverse cell mapping failed."));
    CapturePoint result;
    for (unsigned int d=0; d<3; ++d) {
        AssertThrow(std::isfinite(r[d]), dealii::ExcMessage("Nonfinite capture coordinates."));
        result.xi[d] = 0.5*(r[d]+1.0);
    }
    result.z = x[2];
    return result;
}

// Restrict every aquifer step to the current cell. The inverse is deliberately
// not clamped: clamping an outside endpoint would hide the exit.
template<int dim>
double capture_cell_fraction(const CellVelocityCacheRT0Split3D<dim> &cache,
                             const dealii::Point<dim> &a, const dealii::Point<dim> &b) {
    if (capture_inside(capture_point(cache,b))) return 1.0;
    double lo=0.0, hi=1.0;
    for (unsigned int i=0; i<40; ++i) {
        const double mid=0.5*(lo+hi);
        if (capture_inside(capture_point(cache,a+mid*(b-a)))) lo=mid;
        else hi=mid;
    }
    return lo;
}

template<int dim>
CaptureHit capture_physical_segment(const WellCaptureModel &model,
    const CellVelocityCacheRT0Split3D<dim> &cache,
    const dealii::Point<dim> &a, const dealii::Point<dim> &b,
    const int subdivisions, const double limit) {
    CaptureHit hit;
    // The starting point has already been tested before the velocity step.
    for (int i = 0; i < subdivisions; ++i) {
        const double fraction = static_cast<double>(i + 1) / subdivisions;
        if (fraction > limit) break;
        const auto point = capture_point(cache, a + fraction * (b - a));
        hit.well = model.at(point);
        if (hit.well >= 0) {
            hit.fraction = fraction;
            break;
        }
    }
    return hit;
}
}

template<int dim>
npsat_trace::WellCaptureModel &NPSAT_TRACE<dim>::get_or_build_well_capture(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    npsat_trace::CellVelocityCacheRT0Split3D<dim> &velocity,
    const unsigned int flow_step) {
    const unsigned int slot=cell->user_index();
    const auto found=well_capture_cache.find(slot);
    if (found != well_capture_cache.end()) return found->second;
    auto &model=well_capture_cache[slot];
    if (slot >= slot_cell_well_links.size()) return model;
    const auto &links=slot_cell_well_links[slot];
    const std::string cell_id = cell->id().to_string();
    for (const auto &link : links) {
        const npsat_trace::FlowKey key{cell_id, link.well_global_index};
        const auto flow=flows_by_cell_well.find(key);
        AssertThrow(flow != flows_by_cell_well.end(),
                    ExcMessage("Well capture: linked cell/well has no flow record"
                               " (rank=" + std::to_string(my_rank) +
                               ", step=" + std::to_string(flow_step) +
                               ", CellId=" + key.cell_id +
                               ", well_global_index=" + std::to_string(key.well_global_index) +
                               ", Eid=" + std::to_string(link.eid) + ")."));
        const double q=flow->second.Qe; // already direction-adjusted
        AssertThrow(std::isfinite(q),ExcMessage("Nonfinite well exchange."));
        if (q >= -1.e-12) continue;
        const double bottom = std::max(link.wbot, link.w_zbot);
        const double top = std::min(link.wtop, link.w_zbot + link.sl);
        AssertThrow(bottom <= top, ExcMessage("Well capture: invalid screen intersection."));
        const auto screen_bottom = npsat_trace::capture_point(velocity, Point<dim>(link.wx, link.wy, bottom));
        const auto screen_top = npsat_trace::capture_point(velocity, Point<dim>(link.wx, link.wy, top));
        for (const auto &endpoint : {screen_bottom, screen_top})
            for (unsigned int d = 0; d < 2; ++d)
                AssertThrow(endpoint.xi[d] >= -1.e-8 && endpoint.xi[d] <= 1.0 + 1.e-8,
                            ExcMessage("Well capture: invalid screen intersection."));
        const npsat_trace::CaptureWell well{link.well_global_index, screen_bottom, screen_top, -q};
        model.wells.push_back(well);
    }
    if (model.wells.empty()) return model;
    // Stable tie ordering independent of map/file ordering.
    std::sort(model.wells.begin(),model.wells.end(),
              [](const npsat_trace::CaptureWell &a,const npsat_trace::CaptureWell &b){return a.id<b.id;});
    std::array<std::array<Point<dim>,4>,6> centers;
    std::array<std::array<double,4>,6> areas;
    std::array<unsigned int,6> counts;
    velocity.get_canonical_outer_subface_geometry(centers,areas,counts);
    const auto &normal=velocity.get_face_subface_normal_velocities();
    for (unsigned int f=0; f<6; ++f) {
        double q=0;
        for (unsigned int k=0;k<counts[f];++k) q+=normal[f][k]*areas[f][k]*topt.sim_opt.direction;
        AssertThrow(std::isfinite(q),ExcMessage("Nonfinite cell face flow."));
        model.outward[f]=std::max(q,0.0);
    }
    return model;
}
#endif
