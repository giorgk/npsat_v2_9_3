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

inline bool capture_inside(const CapturePoint &p) {
    for (double v : p.xi) if (v < 0.0 || v > 1.0) return false;
    return true;
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

struct MappedCapturePoint { CapturePoint point; double fraction; };

template<int dim>
void append_capture_chords(const CellVelocityCacheRT0Split3D<dim> &cache,
                          const dealii::Point<dim> &a, const dealii::Point<dim> &b,
                          const CapturePoint &ra, const CapturePoint &rb,
                          double ta, double tb, unsigned int depth,
                          std::vector<MappedCapturePoint> &points) {
    const auto middle = 0.5*(a+b);
    const auto rm = capture_point(cache,middle);
    double error=0.0;
    for (unsigned int d=0; d<3; ++d)
        error=std::max(error,std::abs(rm.xi[d]-0.5*(ra.xi[d]+rb.xi[d])));
    if (error > 1.e-7 && depth < 16) {
        append_capture_chords(cache,a,middle,ra,rm,ta,0.5*(ta+tb),depth+1,points);
        append_capture_chords(cache,middle,b,rm,rb,0.5*(ta+tb),tb,depth+1,points);
    } else {
        AssertThrow(error <= 1.e-7, dealii::ExcMessage("Well capture: distorted-cell mapping unresolved."));
        points.push_back({rb,tb});
    }
}

template<int dim>
std::vector<MappedCapturePoint> capture_chords(
    const CellVelocityCacheRT0Split3D<dim> &cache,
    const dealii::Point<dim> &a, const dealii::Point<dim> &b) {
    std::vector<MappedCapturePoint> result;
    const auto ra=capture_point(cache,a), rb=capture_point(cache,b);
    result.push_back({ra,0.0});
    append_capture_chords(cache,a,b,ra,rb,0.0,1.0,0,result);
    return result;
}

template<int dim>
CaptureHit capture_physical_segment(const WellCaptureModel &model,
    const CellVelocityCacheRT0Split3D<dim> &cache,
    const dealii::Point<dim> &a, const dealii::Point<dim> &b) {
    if (model.wells.empty()) return CaptureHit();
    const auto points=capture_chords(cache,a,b);
    for (unsigned int k=1; k<points.size(); ++k) {
        auto hit=model.intersect(points[k-1].point,points[k].point);
        if (hit.well >= 0) {
            hit.fraction=points[k-1].fraction + hit.fraction*(points[k].fraction-points[k-1].fraction);
            return hit;
        }
    }
    return CaptureHit();
}
}

template<int dim>
npsat_trace::WellCaptureModel &NPSAT_TRACE<dim>::get_or_build_well_capture(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    npsat_trace::CellVelocityCacheRT0Split3D<dim> &velocity) {
    const unsigned int slot=cell->user_index();
    const auto found=well_capture_cache.find(slot);
    if (found != well_capture_cache.end()) return found->second;
    auto &model=well_capture_cache[slot];
    if (slot >= slot_cell_well_links.size()) return model;
    const auto &links=slot_cell_well_links[slot];
    std::vector<std::pair<npsat_trace::CellWellLink,double> > sources;
    for (const auto &link : links) {
        const npsat_trace::FlowKey key{cell->active_cell_index(),link.well_global_index};
        const auto flow=flows_by_cell_well.find(key);
        AssertThrow(flow != flows_by_cell_well.end(),
                    ExcMessage("Well capture: linked cell/well has no flow record."));
        const double q=flow->second.Qe; // already direction-adjusted
        AssertThrow(std::isfinite(q),ExcMessage("Nonfinite well exchange."));
        if (q > 1.e-12) sources.push_back({link,q});
        if (q >= -1.e-12) continue;
        const auto r=npsat_trace::capture_point(velocity,
            Point<dim>(link.wx,link.wy,link.w_zbot+0.5*link.sl));
        npsat_trace::CaptureWell well;
        well.id=link.well_global_index;
        well.u=r.xi[0]; well.v=r.xi[1];
        well.bottom=std::max(link.wbot,link.w_zbot);
        well.top=std::min(link.wtop,link.w_zbot+link.sl);
        well.rate=-q; well.lambda=1.0;
        AssertThrow(well.bottom <= well.top && well.u >= -1.e-8 && well.u <= 1+1.e-8 &&
                    well.v >= -1.e-8 && well.v <= 1+1.e-8,
                    ExcMessage("Well capture: invalid screen intersection."));
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
    std::vector<std::pair<Point<dim>,double> > seeds;
    constexpr unsigned int nq=4;
    for (unsigned int f=0; f<6; ++f) {
        double q=0;
        for (unsigned int k=0;k<counts[f];++k) q+=normal[f][k]*areas[f][k]*topt.sim_opt.direction;
        AssertThrow(std::isfinite(q),ExcMessage("Nonfinite cell face flow."));
        model.outward[f]=std::max(q,0.0);
        if (q >= -1.e-12) continue;
        model.inflow-=q;
        const unsigned int axis=f/2, d1=(axis+1)%3, d2=(axis+2)%3;
        std::vector<std::pair<Point<dim>,double> > face_seeds;
        double sum_area=0;
        for (unsigned int i=0;i<nq;++i) for (unsigned int j=0;j<nq;++j) {
            Point<dim> r;
            r[axis]=f%2; r[d1]=(i+0.5)/nq; r[d2]=(j+0.5)/nq;
            Point<dim> minus=r, plus=r;
            const double h=1.e-5;
            minus[d1]-=h; plus[d1]+=h;
            const auto tangent1=(velocity.capture_physical_point(plus)-velocity.capture_physical_point(minus))/(2*h);
            minus=r; plus=r; minus[d2]-=h; plus[d2]+=h;
            const auto tangent2=(velocity.capture_physical_point(plus)-velocity.capture_physical_point(minus))/(2*h);
            const double area=cross_product_3d(tangent1,tangent2).norm();
            sum_area+=area;
            r[axis]=(f%2) ? 1.0-1.e-8 : 1.e-8;
            face_seeds.push_back({velocity.capture_physical_point(r),area});
        }
        AssertThrow(sum_area>0,ExcMessage("Degenerate capture inflow face."));
        for (const auto &seed:face_seeds) seeds.push_back({seed.first,-q*seed.second/sum_area});
    }
    // Well sources are available in Qe but their subcell release distribution
    // is not. Use uniform screen-height/azimuth quadrature around the axis.
    for (const auto &source:sources) {
        const auto &link=source.first;
        const double bottom=std::max(link.wbot,link.w_zbot);
        const double top=std::min(link.wtop,link.w_zbot+link.sl);
        const auto axis=npsat_trace::capture_point(velocity,Point<dim>(link.wx,link.wy,0.5*(bottom+top)));
        model.inflow+=source.second;
        for(unsigned int i=0;i<nq;++i) for(unsigned int j=0;j<nq;++j) {
            const double angle=2*std::acos(-1.0)*(j+0.5)/nq;
            Point<dim> r;
            r[0]=std::max(1.e-8,std::min(1.0-1.e-8,axis.xi[0]+1.e-6*std::cos(angle)));
            r[1]=std::max(1.e-8,std::min(1.0-1.e-8,axis.xi[1]+1.e-6*std::sin(angle)));
            r[2]=0.5;
            auto x=velocity.capture_physical_point(r);
            x[2]=bottom+(top-bottom)*(i+0.5)/nq;
            seeds.push_back({x,source.second/(nq*nq)});
        }
    }
    double out=0; for(double q:model.outward) out+=q;
    for(const auto &w:model.wells) out+=w.rate;
    model.relative_budget_error=std::abs(model.inflow-out)/std::max(1.e-12,std::max(model.inflow,out));
    // Avoid doing any trajectory work when the available budget cannot close.
    if (model.inflow>1.e-12 && model.relative_budget_error<=0.01) {
        std::vector<npsat_trace::CapturePath> paths;
        const double water_table = slot < slot_water_table_elevation.size()
            ? slot_water_table_elevation[slot] : std::numeric_limits<double>::quiet_NaN();
        for (const auto &seed:seeds) {
            npsat_trace::CapturePath path;
            path.weight=seed.second;
            Point<dim> x=seed.first;
            if (std::isfinite(water_table) && x[2]>water_table+1.e-10) {
                // Do not silently redistribute flux seeded into a dry region.
                paths.push_back(std::move(path));
                continue;
            }
            path.points.push_back(npsat_trace::capture_point(velocity,x));
            for (unsigned int step=0;step<256;++step) {
                Point<dim> reference;
                velocity.get_clamped_ref_coords(x,reference);
                Tensor<1,dim> u; double speed;
                velocity.compute_velocity_at_particle(x,reference,npsat_trace::VelocityInterpolationScheme::coarse_rt,u,speed);
                if (speed<=1.e-14) break;
                const auto direction=topt.sim_opt.direction*u/speed;
                double distance=velocity.directional_bbox_width(direction)/32.0;
                Point<dim> proposed=x+distance*direction;
                // Midpoint integration with a local velocity-change limit.
                // In particular, do not hop across a coarse-RT stagnation point
                // repeatedly when building the calibration trajectories.
                for (unsigned int refine=0; refine<20; ++refine) {
                    const auto middle=x+0.5*distance*direction;
                    if (!npsat_trace::capture_inside(npsat_trace::capture_point(velocity,middle))) {
                        proposed=x+distance*direction;
                        break;
                    }
                    Point<dim> mr; velocity.get_clamped_ref_coords(middle,mr);
                    Tensor<1,dim> mu; double ms;
                    velocity.compute_velocity_at_particle(middle,mr,
                        npsat_trace::VelocityInterpolationScheme::coarse_rt,mu,ms);
                    if ((mu-u).norm()<=0.25*speed || refine==19) {
                        proposed=x+topt.sim_opt.direction*(distance/speed)*mu;
                        break;
                    }
                    distance*=0.5;
                }
                double fraction=npsat_trace::capture_cell_fraction(velocity,x,proposed);
                if (std::isfinite(water_table) && proposed[2]>=water_table && proposed[2]>x[2])
                    fraction=std::min(fraction,std::max(0.0,(water_table-x[2])/(proposed[2]-x[2])));
                const auto next=x+fraction*(proposed-x);
                const auto chord=npsat_trace::capture_chords(velocity,x,next);
                for(unsigned int k=1;k<chord.size();++k) path.points.push_back(chord[k].point);
                if(fraction<1.0) { path.exited=true; break; }
                if(next.distance(x)<1.e-12*cell->diameter()) break;
                x=next;
            }
            paths.push_back(std::move(path));
        }
        model.calibrate(paths);
    }
    if (!model.calibrated && well_capture_warnings++ < 8)
        std::cerr << "Well capture rank " << my_rank << " cell " << cell->id().to_string()
                  << ": budget residual=" << model.relative_budget_error
                  << ", capture residual=" << model.relative_capture_error
                  << ", unresolved trajectory flow=" << model.unresolved_flow_fraction
                  << (model.relative_budget_error>0.01 || model.inflow<=1.e-12
                      ? "; using uncalibrated lambda=1 (incomplete available budget)."
                      : "; using best bounded fit (target unresolved by local trajectories).")
                  << std::endl;
    return model;
}
#endif
