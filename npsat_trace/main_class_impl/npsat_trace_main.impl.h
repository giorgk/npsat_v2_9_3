//
// Created by giorgk on 7/2/26.
//

#ifndef NPSAT_TRACE_MAIN_IMPL_H
#define NPSAT_TRACE_MAIN_IMPL_H

template <int dim>
void NPSAT_TRACE<dim>::distribute_particles(const std::vector<npsat_trace::ParticleSeed> &seeds0) {
    AssertThrow(global_bounding_boxes.size() == n_proc,
                dealii::ExcMessage("global_bounding_boxes not initialized. Call setup_triangulation_helpers()."));

    // ParticleHandler must have been initialized once:
    // particle_handler.initialize(triangulation, mapping, /*n_properties=*/4);
    AssertThrow(particle_handler.n_properties_per_particle() == npsat_trace::n_particle_props,
                dealii::ExcMessage("ParticleHandler property count does not match npsat_trace::n_particle_props."));

    // Only rank 0 provides input. Other ranks pass empty vectors.
    std::vector<Point<dim>> positions;
    std::vector<std::vector<double>> properties;

    if (my_rank == 0) {
        positions.reserve(seeds0.size());
        properties.reserve(seeds0.size());
        for (const auto &s : seeds0) {
            positions.emplace_back(s.x, s.y, s.z);
            std::vector<double> pr(npsat_trace::n_particle_props);
            pr[npsat_trace::pPid] = 0.0;
            pr[npsat_trace::pEid] = double(s.Eid);
            pr[npsat_trace::pSid] = double(s.Sid);
            //pr[npsat_trace::pRt]  = double(s.rt);
            pr[npsat_trace::pDtRemaining] = 0.0;               // will be set at step start
            pr[npsat_trace::pVmag] = 0.0;
            pr[npsat_trace::pState] = 0.0;
            pr[npsat_trace::pStreamlineSteps] = 0.0;
            pr[npsat_trace::pAge] = 0.0;
            pr[npsat_trace::pBBoxMinX] = npsat_trace::coarse_streamline_coord(s.x);
            pr[npsat_trace::pBBoxMinY] = npsat_trace::coarse_streamline_coord(s.y);
            pr[npsat_trace::pBBoxMinZ] = npsat_trace::coarse_streamline_coord(s.z);
            pr[npsat_trace::pBBoxMaxX] = pr[npsat_trace::pBBoxMinX];
            pr[npsat_trace::pBBoxMaxY] = pr[npsat_trace::pBBoxMinY];
            pr[npsat_trace::pBBoxMaxZ] = pr[npsat_trace::pBBoxMinZ];
            pr[npsat_trace::pNoExpandCount] = 0.0;
            //pr[npsat_trace::pRf]  = double(s.rf);
            properties.push_back(std::move(pr));
        }
    }

    // Collective call: all ranks must enter, even if they contribute no points.
    (void)particle_handler.insert_global_particles(positions, global_bounding_boxes, properties /* ids = {} */);

    // Optional safety/consistency (cheap): keeps internal structures tidy.
    particle_handler.sort_particles_into_subdomains_and_cells();

    const unsigned int n_local_particles = particle_handler.n_locally_owned_particles();
    /*for (unsigned int r = 0; r < n_proc; ++r)
    {
        MPI_Barrier(mpi_communicator);
        std::cout << "Rank " << my_rank
                  << " owns " << n_local_particles
                  << " particles." << std::endl;
        MPI_Barrier(mpi_communicator);
    }*/

}

template<int dim>
void NPSAT_TRACE<dim>::load_data_step(const std::string & file_prefix, unsigned int step) {
    // Load RT0 face-normal velocities for this step
    load_vface_rt0_values_step(file_prefix, step);

    for (unsigned int i = 0; i < all_cells_cache_valid.size(); ++i) {
        all_cells_cache[i].clear();
        all_cells_cache_valid[i] = false;
    }


    read_particle_well_flows_for_step(file_prefix, step);
    read_water_table_for_step(file_prefix, step);
}

template<int dim>
void NPSAT_TRACE<dim>::load_vface_rt0_values_step(const std::string &prefix, const unsigned int step_no)
{
    npsat_trace::read_vface_rt0_vals_into_vector(prefix,
                                                 locally_owned_dofs,
                                                 locally_relevant_dofs,
                                                 my_rank,
                                                 step_no,
                                                 vface);
}

template <int dim>
void NPSAT_TRACE<dim>::read_particle_well_flows_for_step(const std::string &prefix, unsigned int step) {
    const std::string step_s   = Utilities::int_to_string(step, 3);
    const std::string rank_s   = Utilities::int_to_string(my_rank, 4);

    std::ostringstream binname;
    binname << prefix << "_particle_well_flows_rank_" << rank_s << "_step_" << step_s << ".bin";

    std::ifstream in(binname.str(), std::ios::binary);
    if (!in.good())
        throw std::runtime_error("Could not open: " + binname.str());

    // header magic[8] = {'P','W','F','L','O','W','v','1'}
    char magic[8];
    in.read(magic, 8);
    if (!in.good())
        throw std::runtime_error("Failed reading header: " + binname.str());

    const char expect_magic[8] = {'P','W','F','L','O','W','v','1'};
    if (std::memcmp(magic, expect_magic, 8) != 0)
        throw std::runtime_error("Bad magic in: " + binname.str());

    std::uint32_t version = 0;
    std::uint32_t step_u32 = 0;
    npsat_trace::read_pod(in, version);
    npsat_trace::read_pod(in, step_u32);

    if (version != 1)
        throw std::runtime_error("Unsupported particle-well-flow version in: " + binname.str());
    if (step_u32 != static_cast<std::uint32_t>(step))
        throw std::runtime_error("Step mismatch in flow file: expected " + std::to_string(step) +
                                 " got " + std::to_string(step_u32));

    std::uint64_t nrec = 0;
    npsat_trace::read_pod(in, nrec);

    flows_by_cell_well.clear();
    flows_by_cell_well.reserve(static_cast<std::size_t>(nrec));

    for (std::uint64_t i = 0; i < nrec; ++i) {
        npsat_trace::WellFlowRecord r;

        // Matches npsat_v2 particle-well-flow v1 records:
        // uint32 cell_id, uint32 well_global_index, double Qe, double Qwbf_bot,
        // double Qwbf_top.
        npsat_trace::read_pod(in, r.cell_id);

        std::uint32_t wgid = 0;
        npsat_trace::read_pod(in, wgid);
        r.well_global_index = wgid;

        npsat_trace::read_pod(in, r.Qe);
        npsat_trace::read_pod(in, r.Qwbf_bot);
        npsat_trace::read_pod(in, r.Qwbf_top);

        r.Qe       *= topt.sim_opt.direction;
        r.Qwbf_bot *= topt.sim_opt.direction;
        r.Qwbf_top *= topt.sim_opt.direction;

        const npsat_trace::FlowKey key{r.cell_id, r.well_global_index};
        flows_by_cell_well.emplace(key, r);
    }
    //imported_flow_step = step_u32;

    if (!in.good())
        throw std::runtime_error("Read failed (particle-well flows): " + binname.str());
}

template <int dim>
void NPSAT_TRACE<dim>::read_water_table_for_step(const std::string &prefix, unsigned int step) {
    slot_water_table_elevation.assign(slot_cellid.size(), std::numeric_limits<double>::quiet_NaN());

    const std::string step_s = Utilities::int_to_string(step, 3);
    const std::string rank_s = Utilities::int_to_string(my_rank, 4);

    std::ostringstream binname;
    binname << prefix << "_water_table_rank_" << rank_s << "_step_" << step_s << ".bin";

    std::ifstream in(binname.str(), std::ios::binary);
    if (!in.good())
        throw std::runtime_error("Could not open: " + binname.str());

    char magic[8];
    in.read(magic, 8);
    if (!in.good())
        throw std::runtime_error("Failed reading header: " + binname.str());

    const char expect_magic[8] = {'W','T','A','B','L','E','v','2'};
    if (std::memcmp(magic, expect_magic, 8) != 0)
        throw std::runtime_error("Bad magic in: " + binname.str());

    std::uint32_t version = 0;
    std::uint32_t dim_u32 = 0;
    std::uint32_t step_u32 = 0;
    std::uint64_t nrec = 0;

    npsat_trace::read_pod(in, version);
    npsat_trace::read_pod(in, dim_u32);
    npsat_trace::read_pod(in, step_u32);
    npsat_trace::read_pod(in, nrec);

    if (version != 2)
        throw std::runtime_error("Unsupported water-table version in: " + binname.str());
    if (dim_u32 != static_cast<std::uint32_t>(dim))
        throw std::runtime_error("Dimension mismatch in: " + binname.str());
    if (step_u32 != static_cast<std::uint32_t>(step))
        throw std::runtime_error("Step mismatch in water-table file: " + binname.str());

    for (std::uint64_t i = 0; i < nrec; ++i) {
        const std::string cell_id_str = npsat_trace::read_string(in);

        double water_table_elevation = 0.0;
        npsat_trace::read_pod(in, water_table_elevation);

        auto it_slot = cellid_to_slot.find(cell_id_str);
        if (it_slot != cellid_to_slot.end())
            slot_water_table_elevation[it_slot->second] = water_table_elevation;
    }

    if (!in.good())
        throw std::runtime_error("Read failed (water-table file): " + binname.str());
}

template <int dim>
npsat_trace::CellVelocityCacheRT0Split3D<dim> &NPSAT_TRACE<dim>::get_or_build_cell_cache(
    const typename DoFHandler<dim>::active_cell_iterator &cell, std::ofstream &dbg_cell_list) {

    AssertThrow(cell->is_locally_owned(), ExcMessage("get_or_build_cell_cache expects a locally owned cell."));
    AssertThrow(cell->is_active(), ExcMessage("Expected active cell."));

    const unsigned int slot = static_cast<unsigned int>(cell->user_index());
    AssertThrow(slot < all_cells_cache.size(), ExcInternalError());

    if (!all_cells_cache_valid[slot]) {
        // New cache build path:
        // - rt0_map contains static face gid/sign/flag information
        // - vface contains this step's ghosted RT0 face-normal velocities
        all_cells_cache[slot].init_cache(cell, rt0_map, vface, my_rank, topt.misc_opt, dbg_cell_list);
        all_cells_cache_valid[slot] = true;
    }
    return all_cells_cache[slot];
}

template <int dim>
npsat_trace::WellBoreTraceResults<dim> NPSAT_TRACE<dim>::well_bore_flow_trace(
        const typename DoFHandler<dim>::active_cell_iterator &current_cell,
        const Point<dim> &x_in) const {
    static_assert(dim == 3, "apply_well_kick_if_close currently assumes dim==3.");

    // Default result: no well interaction.
    // The caller interprets terminate=false and unchanged position/cell as
    // "continue with normal aquifer velocity tracing".
    npsat_trace::WellBoreTraceResults<dim> R;
    R.new_pos = x_in;
    R.new_cell = current_cell;
    R.terminate = false;
    // If a later branch terminates without setting a more specific reason,
    // it is a normal well capture at the top/bottom of the screened interval.
    R.end_reason = npsat_trace::er_well_captured;

    // Small tolerance used to classify flows as positive, negative, or zero.
    // The well flow files have already been multiplied by trace direction when
    // they were loaded, so the signs here are always in the active tracking
    // direction, regardless of forward/backward mode.
    const double flow_eps = 1.0e-12;

    // Sentinel used by find_link_and_flow(): any well is allowed on the first
    // capture test, but after capture we must keep routing along the same well.
    const std::uint32_t any_well = std::numeric_limits<std::uint32_t>::max();

    // Return the neighbor across a vertical face that contains the nudged
    // particle position.  This hides the same-level/refined-neighbor distinction
    // from the well-routing loop below.
    auto neighbor_that_contains_point =
       [&](const typename DoFHandler<dim>::active_cell_iterator &cell,
           const unsigned int face,
           const Point<dim> &pt) -> typename DoFHandler<dim>::active_cell_iterator {

           // Precondition: !cell->at_boundary(face). Boundary cases are handled
           // before calling this lambda because reaching a screen end captures
           // the particle.
           auto neigh = cell->neighbor(face);
           if (!neigh->has_children()) {
               // Same-level neighbor: should contain the point (up to tolerance), but still check.
               if (neigh->point_inside(pt))
                   return neigh;

               // Fallback: return neighbor anyway (keeps things moving; caller can still use pt).
               return neigh;
           }
           // Refined neighbor: choose the active child that contains the point.
           for (unsigned int c = 0; c < neigh->n_children(); ++c) {
               auto ch = neigh->child(c);
               if (ch->is_active() && ch->point_inside(pt))
                   return ch;
           }
           // If none matched (rare due to tolerance), return the neighbor as fallback.
           // If you have a tolerance-based "point_inside" helper, use it here.
           return neigh;
    };

    // Convert a scalar flow into a routing sign:
    // +1 means upward wellbore movement, -1 means downward movement, and 0
    // means there is no reliable axial drive.
    auto flow_sign = [flow_eps](const double q) -> int {
        if (q > flow_eps)
            return 1;
        if (q < -flow_eps)
            return -1;
        return 0;
    };

    // Map a deal.II cell iterator into the compact per-rank "slot" used by
    // slot_cell_well_links.  The slot map was built from persistent CellId
    // strings when the trace-side mesh/mapping was loaded.
    auto cell_slot = [&](const typename DoFHandler<dim>::active_cell_iterator &cell,
                         unsigned int &slot) -> bool {
        const auto it_slot = cellid_to_slot.find(cell->id().to_string());
        if (it_slot == cellid_to_slot.end())
            return false;
        slot = it_slot->second;
        return slot < slot_cell_well_links.size();
    };

    // Select the well link and adjusted flow record that should control routing
    // in a cell.
    //
    // Two modes are used:
    // 1. require_threshold=true: initial capture decision from the aquifer.
    //    First, a small "definite check" radius is applied to all wells,
    //    regardless of Qe.  If the nearest definite well is attracting
    //    (Qe < 0 after direction adjustment), the particle is captured by it.
    //    If no definite well captures the particle, the wider Qe-based
    //    influence radius is evaluated for all attracting wells.  When several
    //    wells can capture, the selected well maximizes abs(Qe)/distance^2.
    //
    // 2. require_threshold=false with required_well set: continued in-well
    //    routing.  The particle has already been captured by one well, so no
    //    lateral threshold is applied.  We only ask whether the same well in
    //    the current cell still has Qe pulling into the well.  If Qe points
    //    out to the aquifer, routing stops and normal aquifer tracing resumes.
    auto find_link_and_flow =
        [&](const typename DoFHandler<dim>::active_cell_iterator &cell,
            const Point<dim> &x,
            const std::uint32_t required_well,
            const bool require_threshold,
            const npsat_trace::CellWellLink *&bestL,
            npsat_trace::WellFlowRecord &bestF) -> bool {

            unsigned int slot = 0;
            if (!cell_slot(cell, slot))
                return false;

            const auto &links = slot_cell_well_links[slot];
            if (links.empty())
                return false;

            const std::uint64_t cell_id_u64 = static_cast<std::uint64_t>(cell->active_cell_index());
            const npsat_trace::CellWellLink *candidateL = nullptr;
            npsat_trace::WellFlowRecord candidateF{};
            double best_dist = std::numeric_limits<double>::infinity();
            double best_weight = -1.0;
            const double cell_diameter = cell->diameter();
            const double definite_distance = std::min(
                std::max(0.0, topt.sim_opt.well_capture_distance),
                std::max(0.0, topt.sim_opt.well_capture_cell_fraction) * cell_diameter);
            const double influence_cap = std::max(0.0, topt.sim_opt.well_influence_max_cell_fraction) * cell_diameter;
            const double q_scale = std::max(0.0, topt.sim_opt.well_influence_q_scale);

            for (const auto &L : links) {
                // During continued in-well routing we only follow the well that
                // captured the particle.  During initial capture any well link
                // in this cell may be considered.
                if (required_well != any_well && L.well_global_index != required_well)
                    continue;

                // The well link only applies inside the full screened interval
                // stored in the cell-well map.  Outside it, this well has no
                // routing effect in this cell.
                if (x[2] > L.wtop || x[2] < L.wbot)
                    continue;

                // The transient flow record is keyed by the active cell index
                // and the internal well index.  The stored Q values have already
                // been sign-corrected for forward/backward tracking.
                const npsat_trace::FlowKey key{cell_id_u64, static_cast<std::uint32_t>(L.well_global_index)};
                const auto itF = flows_by_cell_well.find(key);
                if (itF == flows_by_cell_well.end())
                    continue;

                const auto &F = itF->second;

                const double rx = x[0] - L.wx;
                const double ry = x[1] - L.wy;
                const double d  = std::sqrt(rx*rx + ry*ry);

                // Continuation mode: no influence radius is used.  A captured
                // particle keeps moving vertically through this same well while
                // Qe continues to pull from aquifer into the well.  If Qe is
                // zero or points from well to aquifer, the particle exits
                // wellbore routing in this cell.
                if (required_well != any_well && !require_threshold) {
                    if (F.Qe >= -flow_eps)
                        return false;
                    bestL = &L;
                    bestF = F;
                    return true;
                }

                // Initial capture, phase 1: definite near-well check.  This
                // radius is min(user distance, user fraction * cell diameter).
                // It is intentionally independent of Qe and of other wells.
                if (require_threshold) {
                    if (d <= definite_distance && d < best_dist) {
                        candidateL = &L;
                        candidateF = F;
                        best_dist = d;
                        best_weight = std::numeric_limits<double>::infinity();
                    }
                    continue;
                }

                // This branch is not used by the current caller, but keeps the
                // helper well-defined if later used to find the nearest
                // attracting well without applying thresholds.
                if (F.Qe >= -flow_eps)
                    continue;

                if (d < best_dist) {
                    candidateL = &L;
                    candidateF = F;
                    best_dist = d;
                }
            }

            if (require_threshold) {
                // If a well was inside the definite radius, it gets priority.
                // It still captures only if its adjusted Qe is into the well.
                if (candidateL) {
                    if (candidateF.Qe >= -flow_eps)
                        return false;
                    bestL = candidateL;
                    bestF = candidateF;
                    return true;
                }

                // Initial capture, phase 2: no well was close enough for the
                // definite check, so evaluate Qe-based influence distances for
                // all attracting wells in this cell.  Each influence radius is
                // capped by a user fraction of cell->diameter().
                for (const auto &L : links) {
                    if (x[2] > L.wtop || x[2] < L.wbot)
                        continue;

                    const npsat_trace::FlowKey key{cell_id_u64, static_cast<std::uint32_t>(L.well_global_index)};
                    const auto itF = flows_by_cell_well.find(key);
                    if (itF == flows_by_cell_well.end())
                        continue;

                    const auto &F = itF->second;
                    if (F.Qe >= -flow_eps)
                        continue;

                    const double rx = x[0] - L.wx;
                    const double ry = x[1] - L.wy;
                    const double d = std::sqrt(rx*rx + ry*ry);
                    const double r_inf = std::min(q_scale * std::sqrt(std::abs(F.Qe)), influence_cap);
                    if (r_inf <= flow_eps)
                        continue;
                    if (d > r_inf)
                        continue;

                    // If several wells overlap, select the strongest pull by
                    // an inverse-square weighting.  flow_eps prevents division
                    // by zero when a particle is exactly on the well axis.
                    const double d_eff = std::max(d, flow_eps);
                    const double weight = std::abs(F.Qe) / (d_eff * d_eff);
                    if (weight > best_weight) {
                        best_weight = weight;
                        candidateL = &L;
                        candidateF = F;
                    }
                }
                // A candidate survived the Qe influence test and won the
                // inverse-square competition.
                if (candidateL) {
                    bestL = candidateL;
                    bestF = candidateF;
                    return true;
                }
            }
            return false;
    };

    // Decide whether a captured particle should move to the segment above or
    // below.  Qwbf_bot and Qwbf_top are vertical wellbore flows at the bottom
    // and top of the current segment after direction correction.
    auto axial_direction = [&](const npsat_trace::CellWellLink &L,
                               const npsat_trace::WellFlowRecord &F,
                               const Point<dim> &x,
                               int &end_reason) -> int {

        const int bot_s = flow_sign(F.Qwbf_bot);
        const int top_s = flow_sign(F.Qwbf_top);

        // Same nonzero sign at both ends means the whole segment is routed in
        // one axial direction.  A zero at one end inherits the nonzero end.
        if (bot_s != 0 && bot_s == top_s)
            return bot_s;
        if (bot_s != 0 && top_s == 0)
            return bot_s;
        if (bot_s == 0 && top_s != 0)
            return top_s;

        // Bottom upward and top downward means both axial flows point toward
        // this segment while Qe is also pulling into the well.  That is a local
        // mass-balance inconsistency for routing, so terminate with a diagnostic
        // flag instead of making an arbitrary move.
        if (bot_s > 0 && top_s < 0) {
            end_reason = npsat_trace::er_well_mass_balance;
            return 0;
        }

        // Bottom downward and top upward means axial flows leave the segment on
        // both ends.  Use a linear interpolation at the particle's z to choose
        // which side of the internal divide the particle belongs to.
        if (bot_s < 0 && top_s > 0) {
            const double seg_bot = L.w_zbot;
            const double seg_top = L.w_zbot + L.sl;
            const double denom = seg_top - seg_bot;
            const double alpha = (std::abs(denom) > 0.0)
                ? npsat_trace::clamp_((x[2] - seg_bot) / denom, 0.0, 1.0)
                : 0.5;
            const double q_interp = F.Qwbf_bot + alpha * (F.Qwbf_top - F.Qwbf_bot);
            return flow_sign(q_interp);
        }

        return 0;
    };

    // Initial capture attempt from the current aquifer cell.  If no well can
    // capture the particle, return the default result and the caller will
    // continue normal velocity-based tracing.
    const npsat_trace::CellWellLink *L = nullptr;
    npsat_trace::WellFlowRecord F{};
    if (!find_link_and_flow(current_cell, x_in, any_well, true, L, F))
        return R;

    // Once captured, follow the same well vertically through adjacent cells.
    // This routing is instantaneous: x/y are preserved and no particle time or
    // age is advanced in this function.  Only z and cell ownership may change.
    typename DoFHandler<dim>::active_cell_iterator cell = current_cell;
    Point<dim> x = x_in;
    std::uint32_t well_id = L->well_global_index;

    // Guard against accidental infinite loops if mesh/well data are malformed.
    // This is the global number of intersected cells for the captured well,
    // plus one final hop for exiting to aquifer/ghost/end.
    std::size_t max_hops_size = static_cast<std::size_t>(L->n_segments) + 1;
    if (max_hops_size < 1)
        max_hops_size = 1;
    const unsigned int max_hops = static_cast<unsigned int>(max_hops_size);

    for (unsigned int hop = 0; hop < max_hops; ++hop) {
        // If routing has entered a ghost/foreign cell, stop here.  The caller
        // will break local tracing and ParticleHandler migration will transfer
        // the particle to the owning processor.
        if (!cell->is_locally_owned()) {
            R.new_pos = x;
            R.new_cell = cell;
            return R;
        }

        // In the current cell, continue only if this same well still pulls the
        // particle from aquifer into well (Qe < 0).  If Qe points toward the
        // aquifer, the particle leaves well routing and resumes normal tracing.
        if (!find_link_and_flow(cell, x, well_id, false, L, F)) {
            R.new_pos = x;
            R.new_cell = cell;
            return R;
        }

        // Determine the next vertical direction inside the captured well
        // segment, or identify a mass-balance/capture stop condition.
        int end_reason = npsat_trace::er_well_captured;
        const int move_sign = axial_direction(*L, F, x, end_reason);

        // Routing inconsistency: terminate and report the mass-balance flag.
        if (end_reason == npsat_trace::er_well_mass_balance) {
            R.new_pos = x;
            R.new_cell = cell;
            R.terminate = true;
            R.end_reason = end_reason;
            return R;
        }

        // No usable axial drive: keep the particle at its current x/y/z and
        // classify it as captured by the well.
        if (move_sign == 0) {
            R.new_pos = x;
            R.new_cell = cell;
            R.terminate = true;
            R.end_reason = npsat_trace::er_well_captured;
            return R;
        }

        const bool to_top = move_sign > 0;
        const unsigned int face = to_top ? 5u : 4u;

        // Use the average vertical edge length to nudge the routed particle
        // just across the top/bottom face.  This helps deal.II place it in the
        // neighboring cell rather than leaving it exactly on a shared face.
        double mean_h = 0.0;
        const unsigned int n_pairs = 4;
        for (unsigned int i = 0; i < n_pairs; ++i) {
            const Point<dim> &pb = cell->vertex(i);
            const Point<dim> &pt = cell->vertex(i + 4);
            mean_h += pb.distance(pt);
        }
        mean_h /= static_cast<double>(n_pairs);
        const double dz_nudge = 0.01 * mean_h;

        const double seg_bot = L->w_zbot;
        const double seg_top = L->w_zbot + L->sl;

        // If the next vertical move exits the global screened interval, the
        // particle has reached the end of the well and is captured.
        const bool at_screen_end = to_top
            ? (seg_top >= L->wtop - dz_nudge)
            : (seg_bot <= L->wbot + dz_nudge);

        // Vertical-only move: preserve x/y exactly.  This is intentional; the
        // capture logic used well distance to decide routing, but the routed
        // particle is not snapped laterally to the well axis.
        x[2] = to_top ? (seg_top + dz_nudge) : (seg_bot - dz_nudge);

        // Boundary or screen end means the route cannot continue to another
        // screened cell, so terminate as well captured.
        if (at_screen_end || cell->at_boundary(face)) {
            R.new_pos = x;
            R.new_cell = cell;
            R.terminate = true;
            R.end_reason = npsat_trace::er_well_captured;
            return R;
        }

        // Move to the active neighbor containing the nudged point and continue
        // the instantaneous well route in that cell.
        cell = neighbor_that_contains_point(cell, face, x);
    }

    // Exceeding max_hops indicates inconsistent cell-well connectivity or a
    // cycle in the route.  Treat it as a mass-balance/routing diagnostic.
    R.new_pos = x;
    R.new_cell = cell;
    R.terminate = true;
    R.end_reason = npsat_trace::er_well_mass_balance;
    return R;
}

#endif //NPSAT_TRACE_MAIN_IMPL_H
