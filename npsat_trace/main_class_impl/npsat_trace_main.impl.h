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
            //pr[npsat_trace::pRf]  = double(s.rf);
            properties.push_back(std::move(pr));
        }
    }

    // Collective call: all ranks must enter, even if they contribute no points.
    (void)particle_handler.insert_global_particles(positions, global_bounding_boxes, properties /* ids = {} */);

    // Optional safety/consistency (cheap): keeps internal structures tidy.
    particle_handler.sort_particles_into_subdomains_and_cells();

    const unsigned int n_local_particles = particle_handler.n_locally_owned_particles();
    for (unsigned int r = 0; r < n_proc; ++r)
    {
        MPI_Barrier(mpi_communicator);
        std::cout << "Rank " << my_rank
                  << " owns " << n_local_particles
                  << " particles." << std::endl;
        MPI_Barrier(mpi_communicator);
    }

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

        // IMPORTANT: This assumes m.cell_id was written as uint64_t in npsat_v2.
        // If the writer uses uint32_t, then:
        //   - change WellFlowRecord::cell_id to uint32_t
        //   - change FlowKey::cell_id accordingly
        npsat_trace::read_pod(in, r.cell_id);

        std::uint32_t wgid = 0;
        npsat_trace::read_pod(in, wgid);
        r.well_global_index = wgid;

        npsat_trace::read_pod(in, r.Qe);
        npsat_trace::read_pod(in, r.Qwbf_bot);
        npsat_trace::read_pod(in, r.Qwbf_top);

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
    const typename DoFHandler<dim>::active_cell_iterator &cell) {

    AssertThrow(cell->is_locally_owned(), ExcMessage("get_or_build_cell_cache expects a locally owned cell."));
    AssertThrow(cell->is_active(), ExcMessage("Expected active cell."));

    const unsigned int slot = static_cast<unsigned int>(cell->user_index());
    AssertThrow(slot < all_cells_cache.size(), ExcInternalError());

    if (!all_cells_cache_valid[slot]) {
        // New cache build path:
        // - rt0_map contains static face gid/sign/flag information
        // - vface contains this step's ghosted RT0 face-normal velocities
        all_cells_cache[slot].init_cache(cell, rt0_map, vface);
        all_cells_cache_valid[slot] = true;
    }
    return all_cells_cache[slot];
}

template <int dim>
npsat_trace::WellBoreTraceResults<dim> NPSAT_TRACE<dim>::well_bore_flow_trace(
        const typename DoFHandler<dim>::active_cell_iterator &current_cell,
        const Point<dim> &x_in) const {
    static_assert(dim == 3, "apply_well_kick_if_close currently assumes dim==3.");

    npsat_trace::WellBoreTraceResults<dim> R;
    R.new_pos = x_in;
    R.new_cell = current_cell;
    R.terminate = false;

    const std::string cell_id_str = current_cell->id().to_string();
    const auto it_slot = cellid_to_slot.find(cell_id_str);
    if (it_slot == cellid_to_slot.end())
        return R;

    const unsigned int slot = it_slot->second;
    if (slot >= slot_cell_well_links.size())
        return R;

    const auto &links = slot_cell_well_links[slot];
    if (links.empty())
        return R;

    const double cell_diam = 0.45* current_cell->diameter();

    // Compute average vertical edge length: pairs (0-4), (1-5), (2-6), (3-7)
    double mean_h = 0.0;
    constexpr unsigned int n_pairs = 4;
    for (unsigned int i = 0; i < n_pairs; ++i)
    {
        const Point<dim> &pb = current_cell->vertex(i);       // bottom vertex
        const Point<dim> &pt = current_cell->vertex(i + 4);   // corresponding top vertex

        mean_h += pb.distance(pt);
    }

    mean_h /= static_cast<double>(n_pairs);
    // 1% vertical nudge
    const double dz_nudge = 0.01 * mean_h;

    // ---- Find closest well within an influence radius (scaled by |Qe|) ----
    // Flow lookup key: assumes your flow files use cell_id == active_cell_index().
    // If not, replace with the exact id you wrote in npsat_v2.
    const std::uint64_t cell_id_u64 = static_cast<std::uint64_t>(current_cell->active_cell_index());

    const npsat_trace::CellWellLink *bestL = nullptr;
    npsat_trace::WellFlowRecord bestF{};
    double best_dist = std::numeric_limits<double>::infinity();
    //double best_rinf = 0.0;

    for (const auto &L : links) {
        // Vertical gate FIRST: if particle is above wtop or below wbot => no effect at all.
        if (x_in[2] > L.wtop || x_in[2] < L.wbot)
            continue;

        const npsat_trace::FlowKey key{cell_id_u64, static_cast<std::uint32_t>(L.well_global_index)};
        const auto itF = flows_by_cell_well.find(key);
        if (itF == flows_by_cell_well.end())
            continue; // no flow data for this cell-well link in this step

        const auto &F = itF->second;

        // If Qe > 0: per your simplification, ignore well entirely (normal field tracing)
        if (F.Qe > 0.0)
            continue;

        const double rx = x_in[0] - L.wx;
        const double ry = x_in[1] - L.wy;
        const double d  = std::sqrt(rx*rx + ry*ry);

        // Influence radius: grows with |Qe|, but NEVER exceeds cell diameter.
        const double Qe_abs = std::abs(F.Qe);

        // Tunables (members):
        //   well_influence_r0: base radius [L]
        //   well_influence_k : scale [L / sqrt(Q)]
        double well_influence_r0   = 50.0;   // base influence radius (m)
        double well_influence_k    = 0.0;
        double r_inf = std::max(0.0, well_influence_r0 + well_influence_k * std::sqrt(Qe_abs));
        r_inf = std::min(r_inf, cell_diam);

        if (d <= r_inf && d < best_dist) {
            bestL = &L;
            bestF = F;
            best_dist = d;
            //best_rinf = r_inf;
        }
    }

    if (!bestL)
        return R;

    // At this point: particle is "captured" by the well (within lateral influence AND within [wbot,wtop])

    // ---- Move within well ONLY (Qe <= 0 guaranteed here) ----
    // If Qe < 0 the well is receiving water from aquifer: particle can only move within the well.
    // Rule: Qwbf_top > 0 => upward movement; else downward (if there is drive).
    //
    // NOTE: This implements a discrete jump to adjacent cell in z, as you requested earlier.
    // If you prefer a continuous z-step inside the same cell, replace move_to_neighbor with z-update.

    const double Qwbf_top = bestF.Qwbf_top;
    const double Qwbf_bot = bestF.Qwbf_bot;

    const bool go_up = (Qwbf_top > 0.0);
    const bool go_down = (!go_up) && (Qwbf_bot < 0.0); // conservative: requires some downward "drive"

    auto neighbor_that_contains_point =
        [&](const typename DoFHandler<dim>::active_cell_iterator &cell,
            const unsigned int face,
            const Point<dim> &pt) -> typename DoFHandler<dim>::active_cell_iterator
    {
        // Precondition: !cell->at_boundary(face)
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
    auto move_to_neighbor = [&](bool to_top) -> void {
        const unsigned int face = to_top ? 5u : 4u; // z+ : z-

        // Snap particle laterally onto well axis when in-well.
        Point<dim> pt = x_in;
        pt[0] = bestL->wx;
        pt[1] = bestL->wy;

        // Nudge across the face by 1% of cell height.
        pt[2] = x_in[2] + (to_top ? +dz_nudge : -dz_nudge);

        // If at boundary: keep position even if outside, and terminate.
        if (current_cell->at_boundary(face))
        {
            R.new_pos = pt;
            R.new_cell = current_cell;
            R.terminate = true;
            return;
        }

        // Otherwise: pick neighbor / child that contains the nudged point.
        auto chosen = neighbor_that_contains_point(current_cell, face, pt);
        R.new_pos  = pt;
        R.new_cell = chosen;

        // Pumped out logic (as you specified): only after capture + intra-well motion reaches wtop.
        // Since we only do intra-well moves here, we can terminate if we crossed wtop.
        if (R.new_pos[2] >= bestL->wtop)
            R.terminate = true;
    };

    if (go_up){
        move_to_neighbor(true);
        return R;
    }

    if (go_down){
        move_to_neighbor(false);
        return R;
    }

    // No axial drive: keep captured but (optionally) snap to well axis; do not move cells.
    // Keep z unchanged (still within [wbot,wtop] by construction here).
    R.new_pos[0] = bestL->wx;
    R.new_pos[1] = bestL->wy;
    R.new_pos[2] = x_in[2];
    R.new_cell   = current_cell;
    return R;




}

#endif //NPSAT_TRACE_MAIN_IMPL_H
