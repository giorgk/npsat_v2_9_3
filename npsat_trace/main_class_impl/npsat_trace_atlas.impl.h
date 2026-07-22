#ifndef NPSAT_V2_TRACE_ATLAS_IMPL_H
#define NPSAT_V2_TRACE_ATLAS_IMPL_H

/**
 * Implementation of trajectory-atlas construction and tracing.
 *
 * MPI ownership rule
 * ------------------
 * A packet is advanced only on the rank owning the active cell containing its
 * current position.  A branch that crosses into a ghost cell is serialized and
 * sent to ghost_cell->subdomain_id().  Atlas objects themselves never cross an
 * MPI boundary: every rank lazily builds atlases only for its locally owned
 * cells from locally relevant face velocities.
 *
 * Branching rule
 * --------------
 * CellTrajectoryAtlas::query() returns fractions whose retained values are
 * normalized to one.  A child receives parent.weight * fraction.  Age and
 * aquifer arc length are advanced with the conditional time and length stored
 * on the cached local trajectories.
 */

template <int dim>
bool NPSAT_TRACE<dim>::find_owned_cell_for_atlas_packet(
    const Point<dim> &position,
    typename DoFHandler<dim>::active_cell_iterator &found) const
{
    for (typename DoFHandler<dim>::active_cell_iterator cell =
             dof_handler_flux.begin_active();
         cell != dof_handler_flux.end(); ++cell)
    {
        if (cell->is_locally_owned() && cell->point_inside(position))
        {
            found = cell;
            return true;
        }
    }
    return false;
}

template <int dim>
unsigned int NPSAT_TRACE<dim>::atlas_owner_for_position(
    const Point<dim> &position) const
{
    // Bounding boxes can overlap at partition boundaries.  The packet is
    // nudged into its destination cell before this function is called, so the
    // first matching rank is deterministic and normally unique.
    for (unsigned int rank = 0; rank < global_bounding_boxes.size(); ++rank)
        for (unsigned int box = 0; box < global_bounding_boxes[rank].size(); ++box)
            if (global_bounding_boxes[rank][box].point_inside(position))
                return rank;
    return n_proc;
}

template <int dim>
void NPSAT_TRACE<dim>::exchange_atlas_packets(
    const std::vector<std::vector<npsat_trace::AtlasPacket<dim> > > &send_packets,
    std::vector<npsat_trace::AtlasPacket<dim> > &received_packets) const
{
    AssertThrow(send_packets.size() == n_proc,
                ExcMessage("Atlas send buffer count must equal MPI size."));

    std::vector<int> send_counts(n_proc, 0), receive_counts(n_proc, 0);
    for (unsigned int rank = 0; rank < n_proc; ++rank)
    {
        AssertThrow(send_packets[rank].size() <=
                        static_cast<std::size_t>(std::numeric_limits<int>::max()),
                    ExcMessage("Too many atlas packets in one MPI message."));
        send_counts[rank] = static_cast<int>(send_packets[rank].size());
    }

    MPI_Alltoall(&send_counts[0], 1, MPI_INT,
                 &receive_counts[0], 1, MPI_INT,
                 mpi_communicator);

    std::vector<int> send_displacements(n_proc, 0), receive_displacements(n_proc, 0);
    for (unsigned int rank = 1; rank < n_proc; ++rank)
    {
        send_displacements[rank] = send_displacements[rank - 1] + send_counts[rank - 1];
        receive_displacements[rank] = receive_displacements[rank - 1] + receive_counts[rank - 1];
    }
    const int n_send = send_displacements.back() + send_counts.back();
    const int n_receive = receive_displacements.back() + receive_counts.back();

    std::vector<npsat_trace::AtlasPacketWire> send_wire(n_send);
    std::vector<npsat_trace::AtlasPacketWire> receive_wire(n_receive);
    for (unsigned int rank = 0; rank < n_proc; ++rank)
        for (unsigned int i = 0; i < send_packets[rank].size(); ++i)
            send_wire[send_displacements[rank] + i] =
                npsat_trace::pack_atlas_packet(send_packets[rank][i]);

    const int wire_bytes = static_cast<int>(sizeof(npsat_trace::AtlasPacketWire));
    std::vector<int> send_bytes(n_proc), receive_bytes(n_proc);
    std::vector<int> send_byte_displacements(n_proc), receive_byte_displacements(n_proc);
    for (unsigned int rank = 0; rank < n_proc; ++rank)
    {
        send_bytes[rank] = send_counts[rank] * wire_bytes;
        receive_bytes[rank] = receive_counts[rank] * wire_bytes;
        send_byte_displacements[rank] = send_displacements[rank] * wire_bytes;
        receive_byte_displacements[rank] = receive_displacements[rank] * wire_bytes;
    }

    MPI_Alltoallv(send_wire.empty() ? 0 : reinterpret_cast<char *>(&send_wire[0]),
                  &send_bytes[0], &send_byte_displacements[0], MPI_BYTE,
                  receive_wire.empty() ? 0 : reinterpret_cast<char *>(&receive_wire[0]),
                  &receive_bytes[0], &receive_byte_displacements[0], MPI_BYTE,
                  mpi_communicator);

    received_packets.clear();
    received_packets.reserve(receive_wire.size());
    for (unsigned int i = 0; i < receive_wire.size(); ++i)
        received_packets.push_back(
            npsat_trace::unpack_atlas_packet<dim>(receive_wire[i]));
}

template <int dim>
npsat_trace::CellTrajectoryAtlas<dim> &
NPSAT_TRACE<dim>::get_or_build_cell_atlas(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    npsat_trace::CellVelocityCacheRT0Split3D<dim> &velocity_cache,
    const unsigned int flow_step,
    std::ofstream &debug_output)
{
    AssertThrow(cell->is_locally_owned(),
                ExcMessage("Only a locally owned cell may build an atlas."));
    const unsigned int slot = static_cast<unsigned int>(cell->user_index());
    AssertIndexRange(slot, all_cell_atlases.size());
    if (!all_cell_atlas_valid[slot] || all_cell_atlas_flow_step[slot] != flow_step)
    {
        build_cell_atlas(cell, velocity_cache, flow_step,
                         all_cell_atlases[slot], debug_output);
        all_cell_atlas_valid[slot] = true;
        all_cell_atlas_flow_step[slot] = flow_step;
    }
    return all_cell_atlases[slot];
}

template <int dim>
void NPSAT_TRACE<dim>::build_cell_atlas(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    npsat_trace::CellVelocityCacheRT0Split3D<dim> &velocity_cache,
    const unsigned int flow_step,
    npsat_trace::CellTrajectoryAtlas<dim> &atlas,
    std::ofstream &debug_output)
{
    static_assert(dim == 3, "Trajectory atlas construction assumes dim == 3.");
    const npsat_trace::Atlas_opt &input = topt.atlas_opt;
    npsat_trace::TrajectoryAtlasOptions options;
    options.quadrature_points_per_subface =
        input.quadrature_points_per_direction * input.quadrature_points_per_direction;
    options.stored_samples_per_trajectory = input.stored_samples_per_trajectory;
    options.maximum_query_samples = input.maximum_query_samples;
    options.kernel_power = input.kernel_power;
    options.kernel_epsilon = input.kernel_epsilon;
    options.flow_tolerance = input.flow_tolerance;
    options.balance_relative_tolerance = input.balance_relative_tolerance;
    options.minimum_branch_fraction = input.minimum_branch_fraction;
    const std::array<double,4> &xv = velocity_cache.get_xv();
    const std::array<double,4> &yv = velocity_cache.get_yv();
    const std::array<double,4> &zb = velocity_cache.get_zb();
    const std::array<double,4> &zt = velocity_cache.get_zt();
    options.metric_scale[0] = std::max(
        *std::max_element(xv.begin(), xv.end()) -
        *std::min_element(xv.begin(), xv.end()), 1.0e-12);
    options.metric_scale[1] = std::max(
        *std::max_element(yv.begin(), yv.end()) -
        *std::min_element(yv.begin(), yv.end()), 1.0e-12);
    const double zmin = std::min(
        *std::min_element(zb.begin(), zb.end()),
        *std::min_element(zt.begin(), zt.end()));
    const double zmax = std::max(
        *std::max_element(zb.begin(), zb.end()),
        *std::max_element(zt.begin(), zt.end()));
    options.metric_scale[2] = std::max(zmax - zmin, 1.0e-12);

    atlas.begin(static_cast<std::uint64_t>(cell->active_cell_index()),
                flow_step, -1, options);

    const std::array<std::array<double, 4>, 6> &normal_velocity =
        velocity_cache.get_face_subface_normal_velocities();
    std::array<std::array<Point<dim>, 4>, 6> canonical_centers;
    std::array<std::array<double, 4>, 6> canonical_areas;
    std::array<unsigned int, 6> canonical_counts;
    velocity_cache.get_canonical_outer_subface_geometry(
        canonical_centers, canonical_areas, canonical_counts);
    std::vector<npsat_trace::AtlasOuterSubfaceData<dim> > subfaces;

    // Build the physical list corresponding to the cache's 24 canonical
    // slots.  Unrefined faces contribute only slot zero; refined parent faces
    // contribute four child faces.
    for (unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
    {
        const unsigned int count = canonical_counts[face_no];
        for (unsigned int q = 0; q < count; ++q)
        {
            npsat_trace::AtlasOuterSubfaceData<dim> data;
            data.face = static_cast<unsigned char>(face_no);
            data.slot = static_cast<unsigned char>(q);
            data.outward_normal_velocity = normal_velocity[face_no][q];
            data.center = canonical_centers[face_no][q];
            data.area = canonical_areas[face_no][q];
            if (count == 1)
            {
                for (unsigned int v = 0; v < 4; ++v)
                    data.vertices[v] = cell->face(face_no)->vertex(v);
            }
            else
            {
                unsigned int physical_child = 0;
                double best = std::numeric_limits<double>::max();
                for (unsigned int child = 0;
                     child < cell->face(face_no)->n_children(); ++child)
                {
                    const double distance = data.center.distance(
                        cell->face(face_no)->child(child)->center());
                    if (distance < best)
                    {
                        best = distance;
                        physical_child = child;
                    }
                }
                for (unsigned int v = 0; v < 4; ++v)
                    data.vertices[v] =
                        cell->face(face_no)->child(physical_child)->vertex(v);
            }
            subfaces.push_back(data);

            npsat_trace::AtlasSubface<dim> stored;
            stored.active = true;
            stored.face = data.face;
            stored.slot = data.slot;
            stored.center = data.center;
            stored.area = data.area;
            stored.outward_flow = data.outward_flow();
            atlas.set_subface(stored);
        }
    }

    const unsigned int cell_slot = static_cast<unsigned int>(cell->user_index());
    const std::uint64_t flow_cell_id = static_cast<std::uint64_t>(cell->active_cell_index());
    if (cell_slot < slot_cell_well_links.size())
        for (unsigned int i = 0; i < slot_cell_well_links[cell_slot].size(); ++i)
        {
            const npsat_trace::CellWellLink &link = slot_cell_well_links[cell_slot][i];
            const npsat_trace::FlowKey key{flow_cell_id, link.well_global_index};
            const typename std::unordered_map<npsat_trace::FlowKey,
                npsat_trace::WellFlowRecord, npsat_trace::FlowKeyHash>::const_iterator found =
                    flows_by_cell_well.find(key);
            if (found != flows_by_cell_well.end())
            {
                npsat_trace::AtlasWellTerminal well;
                well.well_id = link.well_global_index;
                // Flow records are currently direction-adjusted on input.
                well.outward_flow = found->second.Qe * topt.sim_opt.direction;
                atlas.add_well(well);
            }
        }

    std::uint32_t trajectory_id = 0;
    const Point<dim> cell_center = cell->center();
    const unsigned int nq = input.quadrature_points_per_direction;

    for (unsigned int sf = 0; sf < subfaces.size(); ++sf)
    {
        const npsat_trace::AtlasOuterSubfaceData<dim> &origin = subfaces[sf];
        const double origin_flow = origin.outward_flow();
        if (origin_flow <= input.flow_tolerance)
            continue;

        for (unsigned int iu = 0; iu < nq; ++iu)
            for (unsigned int iv = 0; iv < nq; ++iv)
            {
                const double u = (static_cast<double>(iu) + 0.5) / nq;
                const double v = (static_cast<double>(iv) + 0.5) / nq;
                Point<dim> start;
                const double shape[4] = {
                    (1.0-u)*(1.0-v), u*(1.0-v),
                    (1.0-u)*v, u*v
                };
                for (unsigned int vertex = 0; vertex < 4; ++vertex)
                    for (unsigned int d = 0; d < dim; ++d)
                        start[d] += shape[vertex] * origin.vertices[vertex][d];
                const double nudge = 1.0e-8;
                start += nudge * (cell_center - start);

                npsat_trace::AtlasTrajectory<dim> trajectory;
                trajectory.id = trajectory_id++;
                trajectory.downstream_origin = npsat_trace::AtlasOrigin::subface(
                    npsat_trace::atlas_subface_id(origin.face, origin.slot));
                trajectory.represented_flow = origin_flow /
                    static_cast<double>(nq * nq);

                Point<dim> x = start;
                double elapsed = 0.0;
                double length = 0.0;
                npsat_trace::AtlasTrajectorySample<dim> first;
                first.position = x;
                trajectory.samples.push_back(first);
                bool finished = false;

                for (unsigned int local_step = 0;
                     local_step < input.maximum_local_steps; ++local_step)
                {
                    // Apply the same geometric/transient well-capture rule as
                    // point tracing.  The atlas path stops at the local well
                    // terminal; well-bore jumps are not included in aquifer
                    // age or aquifer arc length.
                    const npsat_trace::WellBoreTraceResults<dim> well_event =
                        well_bore_flow_trace(cell, x);
                    if (well_event.terminate || well_event.new_cell != cell ||
                        well_event.new_pos != x)
                    {
                        std::uint32_t nearest_well = 0;
                        double nearest_distance = std::numeric_limits<double>::max();
                        if (cell_slot < slot_cell_well_links.size())
                            for (unsigned int w = 0;
                                 w < slot_cell_well_links[cell_slot].size(); ++w)
                            {
                                const npsat_trace::CellWellLink &link =
                                    slot_cell_well_links[cell_slot][w];
                                if (x[2] < link.wbot || x[2] > link.wtop)
                                    continue;
                                const double dx = x[0] - link.wx;
                                const double dy = x[1] - link.wy;
                                const double distance = dx*dx + dy*dy;
                                if (distance < nearest_distance)
                                {
                                    nearest_distance = distance;
                                    nearest_well = link.well_global_index;
                                }
                            }
                        trajectory.upstream_terminal =
                            npsat_trace::AtlasTerminal::well(nearest_well);
                        finished = true;
                        break;
                    }

                    Point<dim> x_ref;
                    velocity_cache.get_clamped_ref_coords(x, x_ref);
                    Tensor<1,dim> velocity;
                    double speed = 0.0;
                    velocity_cache.compute_velocity_at_particle(
                        x, x_ref, topt.sim_opt.velocity_interpolation,
                        velocity, speed);
                    velocity /= topt.sim_opt.porosity;
                    speed /= topt.sim_opt.porosity;
                    if (speed <= topt.sim_opt.stagnant_velocity_threshold)
                    {
                        trajectory.upstream_terminal =
                            npsat_trace::AtlasTerminal::stagnant();
                        finished = true;
                        break;
                    }

                    const Tensor<1,dim> direction = velocity / speed;
                    const double width = velocity_cache.directional_bbox_width(direction);
                    const double ds = std::max(width / 20.0, 1.0e-10);
                    const double dt = ds / speed;
                    const Point<dim> proposed =
                        x - ds * direction; // atlas construction is backward

                    if (!cell->point_inside(proposed))
                    {
                        const npsat_trace::FindHexExitResult exit =
                            npsat_trace::find_hex_exit(
                                x, proposed,
                                velocity_cache.get_xv(), velocity_cache.get_yv(),
                                velocity_cache.get_zb(), velocity_cache.get_zt());
                        if (!exit.found)
                        {
                            trajectory.upstream_terminal =
                                npsat_trace::AtlasTerminal::unresolved();
                            finished = true;
                            break;
                        }
                        const Point<dim> endpoint = exit.value.intersection_point;
                        const double segment = x.distance(endpoint);
                        elapsed += segment / speed;
                        length += segment;
                        npsat_trace::AtlasTrajectorySample<dim> sample;
                        sample.position = endpoint;
                        sample.backward_time = elapsed;
                        sample.aquifer_length = length;
                        trajectory.samples.push_back(sample);

                        const unsigned int exit_face =
                            static_cast<unsigned int>(exit.value.face_index);
                        unsigned int best_slot = 0;
                        double best_distance = std::numeric_limits<double>::max();
                        for (unsigned int candidate = 0; candidate < subfaces.size(); ++candidate)
                            if (subfaces[candidate].face == exit_face)
                            {
                                const double distance =
                                    endpoint.distance(subfaces[candidate].center);
                                if (distance < best_distance)
                                {
                                    best_distance = distance;
                                    best_slot = subfaces[candidate].slot;
                                }
                            }
                        const unsigned int terminal_id =
                            npsat_trace::atlas_subface_id(exit_face, best_slot);
                        // A valid backward atlas path must end on a forward
                        // inflow.  Loops or interpolation overshoots that
                        // return to a forward outflow are retained explicitly
                        // as unresolved rather than corrupting the transfer
                        // matrix.
                        if (atlas.subfaces()[terminal_id].is_forward_inflow(
                                input.flow_tolerance))
                            trajectory.upstream_terminal =
                                npsat_trace::AtlasTerminal::subface(terminal_id);
                        else
                            trajectory.upstream_terminal =
                                npsat_trace::AtlasTerminal::unresolved();
                        finished = true;
                        break;
                    }

                    x = proposed;
                    elapsed += dt;
                    length += ds;
                    npsat_trace::AtlasTrajectorySample<dim> sample;
                    sample.position = x;
                    sample.backward_time = elapsed;
                    sample.aquifer_length = length;
                    trajectory.samples.push_back(sample);
                }

                if (!finished)
                    trajectory.upstream_terminal =
                        npsat_trace::AtlasTerminal::unresolved();
                trajectory.samples =
                    npsat_trace::resample_atlas_trajectory<dim>(
                        trajectory.samples,
                        input.stored_samples_per_trajectory);
                atlas.add_trajectory(trajectory);
            }
    }

    atlas.finalize();
    if (debug_output.good())
        debug_output << "ATLAS " << cell->id().to_string()
                     << " step " << flow_step
                     << " trajectories " << atlas.trajectories().size()
                     << " Qin " << atlas.total_forward_inflow()
                     << " Qout " << atlas.total_forward_outflow()
                     << " residual " << atlas.mass_balance_error()
                     << " max_origin_relative_error "
                     << atlas.maximum_origin_relative_error() << '\n';
}

template <int dim>
void NPSAT_TRACE<dim>::run_trajectory_atlas()
{
    static_assert(dim == 3, "Trajectory atlas tracing assumes dim == 3.");
    AssertThrow(topt.sim_opt.direction < 0.0,
                ExcMessage("Trajectory-atlas transport currently supports backward tracing only."));

    load_triangulation();
    if (topt.exit_after_load_tria)
        return;
    MPI_Barrier(mpi_communicator);
    setup_triangulation_helpers();
    MPI_Barrier(mpi_communicator);
    setup_system();
    MPI_Barrier(mpi_communicator);

    const std::vector<double> delta_times =
        npsat_trace::read_delta_time_file(topt.delta_time_file);
    AssertThrow(!delta_times.empty(), ExcMessage("No transient time steps were supplied."));

    npsat_trace::ParticleReader reader(topt.n_paticles_parallel);
    if (my_rank == 0)
        reader.open(topt.particles_file);

    unsigned int iteration = 0;
    while (true)
    {
        std::vector<npsat_trace::ParticleSeed> seeds;
        if (my_rank == 0)
            seeds = reader.read_next_chunk();
        int have = (my_rank == 0 && !seeds.empty()) ? 1 : 0;
        MPI_Bcast(&have, 1, MPI_INT, 0, mpi_communicator);
        if (!have)
            break;

        std::vector<std::vector<npsat_trace::AtlasPacket<dim> > > initial_send(n_proc);
        if (my_rank == 0)
            for (unsigned int i = 0; i < seeds.size(); ++i)
            {
                npsat_trace::AtlasPacket<dim> packet;
                packet.receptor_id = seeds[i].Eid;
                packet.seed_id = seeds[i].Sid;
                packet.branch_id = (static_cast<std::uint64_t>(seeds[i].Eid) << 32) ^
                                   static_cast<std::uint64_t>(seeds[i].Sid);
                packet.position = Point<dim>(seeds[i].x, seeds[i].y, seeds[i].z);
                packet.weight = 1.0;
                const unsigned int owner = atlas_owner_for_position(packet.position);
                AssertThrow(owner < n_proc, ExcMessage("Atlas seed lies outside all rank bounding boxes."));
                initial_send[owner].push_back(packet);
            }

        std::vector<npsat_trace::AtlasPacket<dim> > active;
        exchange_atlas_packets(initial_send, active);

        const std::string rank_string = Utilities::int_to_string(my_rank, 4);
        const std::string iteration_string = Utilities::int_to_string(iteration, 4);
        std::ofstream output((topt.output_prefix + "_atlas_rank_" + rank_string +
                              "_iter_" + iteration_string + ".dat").c_str());
        output << "# Eid Sid branch parent weight age length x y z terminal_kind terminal_id\n";

        unsigned int flow_step = static_cast<unsigned int>(delta_times.size() - 1);
        unsigned long long time_pass = 0;
        while (true)
        {
            load_data_step(topt.input_prefix, flow_step);
            for (unsigned int p = 0; p < active.size(); ++p)
            {
                active[p].flow_step = flow_step;
                active[p].dt_remaining = delta_times[flow_step];
                active[p].state = npsat_trace::AtlasPacketState::active;
            }

            std::vector<npsat_trace::AtlasPacket<dim> > completed;
            unsigned int exchange_epoch = 0;
            while (true)
            {
                std::vector<std::vector<npsat_trace::AtlasPacket<dim> > > send(n_proc);
                std::vector<npsat_trace::AtlasPacket<dim> > local_queue;
                local_queue.swap(active);

                while (!local_queue.empty())
                {
                    npsat_trace::AtlasPacket<dim> parent = local_queue.back();
                    local_queue.pop_back();
                    if (parent.dt_remaining <= topt.sim_opt.dt_eps)
                    {
                        parent.state = npsat_trace::AtlasPacketState::step_complete;
                        completed.push_back(parent);
                        continue;
                    }
                    if (topt.sim_opt.max_age >= 0.0 &&
                        parent.age >= static_cast<double>(topt.sim_opt.max_age))
                    {
                        output << parent.receptor_id << ' ' << parent.seed_id << ' '
                               << parent.branch_id << ' ' << parent.parent_branch_id << ' '
                               << parent.weight << ' ' << parent.age << ' '
                               << parent.aquifer_length << ' '
                               << parent.position[0] << ' ' << parent.position[1] << ' '
                               << parent.position[2] << " 3 0\n";
                        continue;
                    }
                    if (parent.generation >= static_cast<unsigned int>(
                            std::max(1, topt.sim_opt.n_max_streamline_steps)))
                    {
                        output << parent.receptor_id << ' ' << parent.seed_id << ' '
                               << parent.branch_id << ' ' << parent.parent_branch_id << ' '
                               << parent.weight << ' ' << parent.age << ' '
                               << parent.aquifer_length << ' '
                               << parent.position[0] << ' ' << parent.position[1] << ' '
                               << parent.position[2] << " 3 0\n";
                        continue;
                    }

                    typename DoFHandler<dim>::active_cell_iterator cell;
                    if (!find_owned_cell_for_atlas_packet(parent.position, cell))
                    {
                        const unsigned int owner = atlas_owner_for_position(parent.position);
                        if (owner < n_proc && owner != my_rank)
                            send[owner].push_back(parent);
                        else
                            output << parent.receptor_id << ' ' << parent.seed_id << ' '
                                   << parent.branch_id << ' ' << parent.parent_branch_id << ' '
                                   << parent.weight << ' ' << parent.age << ' '
                                   << parent.aquifer_length << ' '
                                   << parent.position[0] << ' ' << parent.position[1] << ' '
                                   << parent.position[2] << " 3 0\n";
                        continue;
                    }

                    std::ofstream no_debug;
                    npsat_trace::CellVelocityCacheRT0Split3D<dim> &velocity_cache =
                        get_or_build_cell_cache(cell, no_debug);
                    npsat_trace::CellTrajectoryAtlas<dim> &atlas =
                        get_or_build_cell_atlas(cell, velocity_cache, flow_step, no_debug);
                    npsat_trace::AtlasQueryResult<dim> query =
                        atlas.query(parent.position, parent.dt_remaining);
                    if (!query.valid)
                    {
                        output << parent.receptor_id << ' ' << parent.seed_id << ' '
                               << parent.branch_id << ' ' << parent.parent_branch_id << ' '
                               << parent.weight << ' ' << parent.age << ' '
                               << parent.aquifer_length << ' '
                               << parent.position[0] << ' ' << parent.position[1] << ' '
                               << parent.position[2] << " 3 0\n";
                        continue;
                    }

                    if (query.branches.size() > topt.atlas_opt.maximum_branches_per_packet)
                    {
                        std::sort(query.branches.begin(), query.branches.end(),
                            [](const npsat_trace::AtlasQueryBranch<dim> &a,
                               const npsat_trace::AtlasQueryBranch<dim> &b) {
                                return a.fraction > b.fraction;
                            });
                        query.branches.resize(topt.atlas_opt.maximum_branches_per_packet);
                        double retained = 0.0;
                        for (unsigned int b = 0; b < query.branches.size(); ++b)
                            retained += query.branches[b].fraction;
                        for (unsigned int b = 0; b < query.branches.size(); ++b)
                            query.branches[b].fraction /= retained;
                    }

                    double child_weight = 0.0;
                    for (unsigned int b = 0; b < query.branches.size(); ++b)
                    {
                        const npsat_trace::AtlasQueryBranch<dim> &branch = query.branches[b];
                        npsat_trace::AtlasPacket<dim> child = parent;
                        child.parent_branch_id = parent.branch_id;
                        child.branch_id = npsat_trace::atlas_child_branch_id(
                            parent.branch_id, flow_step, b);
                        child.generation = parent.generation + 1;
                        child.weight = parent.weight * branch.fraction;
                        child.age += branch.advanced_time;
                        child.aquifer_length += branch.advanced_length;
                        child.dt_remaining = std::max(
                            parent.dt_remaining - branch.advanced_time, 0.0);
                        child.position = branch.advanced_position;
                        child_weight += child.weight;

                        if (!branch.reaches_terminal)
                        {
                            child.dt_remaining = 0.0;
                            child.state = npsat_trace::AtlasPacketState::step_complete;
                            completed.push_back(child);
                            continue;
                        }

                        if (branch.terminal.kind == npsat_trace::AtlasTerminalKind::subface)
                        {
                            const unsigned int face =
                                npsat_trace::atlas_face_from_subface_id(branch.terminal.id);
                            const unsigned int subface =
                                npsat_trace::atlas_slot_from_subface_id(branch.terminal.id);
                            if (cell->at_boundary(face))
                            {
                                output << child.receptor_id << ' ' << child.seed_id << ' '
                                       << child.branch_id << ' ' << child.parent_branch_id << ' '
                                       << child.weight << ' ' << child.age << ' '
                                       << child.aquifer_length << ' '
                                       << child.position[0] << ' ' << child.position[1] << ' '
                                       << child.position[2] << " 0 " << branch.terminal.id << '\n';
                                continue;
                            }

                            typename DoFHandler<dim>::cell_iterator neighbor;
                            if (cell->face(face)->has_children())
                                neighbor = cell->neighbor_child_on_subface(face, subface);
                            else
                                neighbor = cell->neighbor(face);
                            const Point<dim> neighbor_center = neighbor->center();
                            child.position += 1.0e-8 * (neighbor_center - child.position);
                            const unsigned int owner = neighbor->subdomain_id();
                            if (owner == my_rank)
                                local_queue.push_back(child);
                            else
                            {
                                AssertThrow(owner < n_proc,
                                            ExcMessage("Atlas branch has invalid destination rank."));
                                send[owner].push_back(child);
                            }
                        }
                        else
                        {
                            output << child.receptor_id << ' ' << child.seed_id << ' '
                                   << child.branch_id << ' ' << child.parent_branch_id << ' '
                                   << child.weight << ' ' << child.age << ' '
                                   << child.aquifer_length << ' '
                                   << child.position[0] << ' ' << child.position[1] << ' '
                                   << child.position[2] << ' '
                                   << static_cast<unsigned int>(branch.terminal.kind) << ' '
                                   << branch.terminal.id << '\n';
                        }
                    }
                    const double weight_scale = std::max(std::abs(parent.weight), 1.0);
                    AssertThrow(std::abs(child_weight - parent.weight) <=
                                    1.0e-10 * weight_scale,
                                ExcMessage("Atlas split did not conserve packet weight."));
                }

                std::vector<npsat_trace::AtlasPacket<dim> > received;
                exchange_atlas_packets(send, received);
                active.swap(received);
                const unsigned long long local_unfinished =
                    static_cast<unsigned long long>(active.size());
                const unsigned long long global_unfinished =
                    Utilities::MPI::sum(local_unfinished, mpi_communicator);
                if (global_unfinished == 0)
                    break;
                ++exchange_epoch;
                if (exchange_epoch >= static_cast<unsigned int>(
                        std::max(1, topt.sim_opt.n_max_proc_exchanges)))
                {
                    for (unsigned int p = 0; p < active.size(); ++p)
                        output << active[p].receptor_id << ' ' << active[p].seed_id << ' '
                               << active[p].branch_id << ' ' << active[p].parent_branch_id << ' '
                               << active[p].weight << ' ' << active[p].age << ' '
                               << active[p].aquifer_length << ' '
                               << active[p].position[0] << ' ' << active[p].position[1] << ' '
                               << active[p].position[2] << " 3 0\n";
                    active.clear();
                    break;
                }
            }

            active.swap(completed);
            const unsigned long long local_active =
                static_cast<unsigned long long>(active.size());
            const unsigned long long global_active =
                Utilities::MPI::sum(local_active, mpi_communicator);
            ++time_pass;
            if (global_active == 0)
                break;
            flow_step = (flow_step + delta_times.size() - 1) % delta_times.size();
        }
        output.close();
        pcout << "Completed atlas particle chunk " << iteration
              << " after " << time_pass << " transient passes." << std::endl;
        ++iteration;
    }
}

#endif // NPSAT_V2_TRACE_ATLAS_IMPL_H
