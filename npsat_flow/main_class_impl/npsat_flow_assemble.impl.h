//
// Created by giorgk on 6/25/26.
//

#ifndef NPSAT_FLOW_ASSEMBLE_IMPL_H
#define NPSAT_FLOW_ASSEMBLE_IMPL_H

template <int dim>
void NPSAT_FLOW<dim>::assemble_system() {
    pcout << "Assembling system at time = " << time_tracking.simulation_step() << " months..." << std::endl;
    double delta_time = time_tracking.duration();

    TimerOutput::Scope t(this->computing_timer, "assemble");

    { // Make system matrices zero
        for (unsigned int i = 0; i < block_system_matrix.n_block_rows(); ++i)
            for (unsigned int j = 0; j < block_system_matrix.n_block_cols(); ++j)
                block_system_matrix.block(i, j) = 0;

        for (unsigned int b = 0; b < block_rhs_vector.n_blocks(); ++b)
            block_rhs_vector.block(b) = 0;
    }

    // Common deal data in assembly
    QGauss<dim> quadrature_formula(fe_flux.degree + 2);
    QGauss<dim-1> face_quadrature_formula(fe_flux.degree + 2);

    FEValues<dim> fe_values_flux(fe_flux, quadrature_formula,
                                 update_values | update_JxW_values | update_gradients | update_quadrature_points);
    FEValues<dim> fe_values_head(fe_head, quadrature_formula,
                                 update_values | update_JxW_values | update_quadrature_points);
    FEFaceValues<dim> fe_face_values_flux(fe_flux, face_quadrature_formula,
                              update_values | update_JxW_values | update_normal_vectors | update_quadrature_points);

    FEFaceValues<dim> fe_face_values_trace(fe_trace, face_quadrature_formula,
                                            update_values | update_JxW_values | update_quadrature_points);

    // Frequently used sizes
    const unsigned int n_q_points        = quadrature_formula.size();
    const unsigned int n_face_q_points   = face_quadrature_formula.size();
    const unsigned int n_flux_dofs       = fe_flux.n_dofs_per_cell();
    const unsigned int n_head_dofs       = fe_head.n_dofs_per_cell();
    const unsigned int n_trace_dofs      = fe_trace.n_dofs_per_cell();

    // Local matrices for each cell
    // A: flux × flux
    FullMatrix<double> local_A(n_flux_dofs, n_flux_dofs);   // A: n_q × n_q
    FullMatrix<double> A_inv(n_flux_dofs, n_flux_dofs);
    // B: head × flux  (divergence block)
    FullMatrix<double> local_B(n_head_dofs, n_flux_dofs);   // B: n_h × n_q
    // C: trace × flux (normal flux on boundary)
    FullMatrix<double> local_C(n_trace_dofs, n_flux_dofs);  // C: n_λ × n_q
    // M: head × head  (storage/accumulation or mass-matrix-like contribution)
    FullMatrix<double> local_M(n_head_dofs, n_head_dofs);   // M: n_h × n_h
    FullMatrix<double> local_GHB(n_trace_dofs, n_trace_dofs);

    // Intermediate Matrices
    FullMatrix<double> E_matrix(n_head_dofs, n_trace_dofs);
    FullMatrix<double> K_inv(n_head_dofs, n_head_dofs);
    FullMatrix<double> S_hat_local(n_trace_dofs, n_trace_dofs);
    FullMatrix<double> ae_matrix(n_head_dofs, n_trace_dofs);

    // RHS contributions
    Vector<double>     local_F(n_head_dofs);     // head RHS        // F: n_h
    Vector<double>     F_native(n_flux_dofs);     // This is used for recovery       // F: n_h
    Vector<double>     local_g(n_trace_dofs);    // trace RHS       // g: n_λ
    Vector<double> M_H_old(n_head_dofs);
    Vector<double> V_vector(n_head_dofs);
    Vector<double> Kinv_V(n_head_dofs);
    Vector<double> b_local(n_trace_dofs);

    // Global–local DOF index buffers
    std::vector<types::global_dof_index> flux_dof_indices(n_flux_dofs);
    std::vector<types::global_dof_index> head_dof_indices(n_head_dofs);
    std::vector<types::global_dof_index> trace_dof_indices(n_trace_dofs);

    const unsigned int n_local_cells = triangulation.n_locally_owned_active_cells();
    std::vector<unsigned char> recharge_receiver(n_local_cells, 0);
    std::vector<double> receiver_recharge_area(n_local_cells, 0.0);
    std::vector<double> receiver_effective_z_top(n_local_cells,std::numeric_limits<double>::quiet_NaN());
    identify_top_active_cells(recharge_receiver, receiver_recharge_area, receiver_effective_z_top);

    double local_recharge_total = 0.0;          // positive into the aquifer
    double local_stream_total = 0.0;            // positive into the aquifer
    double local_well_prescribed_total = 0.0;   // same sign convention as mnwells.pumping_rate()

    const unsigned int n_wells = static_cast<unsigned int>(mnwells.wells.size());
    const double wet_screen_length_tol = 1e-10;
    std::vector<double> local_well_cwc_eff_sum(n_wells, 0.0);
    std::vector<double> local_well_wet_screen_sum(n_wells, 0.0);
    std::vector<double> local_well_total_screen_sum(n_wells, 0.0);


    // =====================================================
    // DEFINE LOCAL MATRICES ACCORDING TO THEORY
    // =====================================================
    // [A   B^T   C^T] [Q]   [0]
    // [B   M/dt   0 ] [H] = [F]
    // [C    0     0 ] [Λ]   [g]
    // =====================================================
    //
    // Where: A: ∫ K⁻¹ φ_i · φ_j dΩ  (n_q × n_q)
    //        B: -∫ ψ_k (∇·φ_i) dΩ  (n_h × n_q)
    //        C: ∫_∂K χ (φ·n) dΓ (n_λ * n_q)
    //        M: ∫ S_s ψ_k ψ_l dΩ  (n_h × n_h)
    //        F = source terms + storage from previous time
    //        g = boundary flux terms

    // Extractor for vector FE (flux)
    FEValuesExtractors::Vector flux(0);

    for (auto trace_cell = dof_handler_trace.begin_active(); trace_cell != dof_handler_trace.end(); ++trace_cell) {
        //pcout << "Cell index: " << cell_index << std::endl;
        if (!trace_cell->is_locally_owned())
            continue;

        const unsigned int slot = trace_cell->user_index();
        Assert(slot != numbers::invalid_unsigned_int, ExcInternalError());
        local_element_data_rt_0dg0.assert_valid_slot(slot);

        // Use the same physical cell to get indices from each handler:
        // typename DoFHandler<dim>::active_cell_iterator flux_cell(&triangulation, trace_cell->level(), trace_cell->index(), &dof_handler_flux);
        // typename DoFHandler<dim>::active_cell_iterator head_cell(&triangulation, trace_cell->level(), trace_cell->index(), &dof_handler_head);
        auto flux_cell = trace_cell->as_dof_handler_iterator(dof_handler_flux);
        auto head_cell = trace_cell->as_dof_handler_iterator(dof_handler_head);

        Assert(flux_cell->active_cell_index() == head_cell->active_cell_index(), ExcInternalError());
        Assert(flux_cell->active_cell_index() == trace_cell->active_cell_index(), ExcInternalError());

        // Reset local matrices
        local_A = 0;
        A_inv = 0;
        local_B = 0;
        local_C = 0;
        local_M = 0;
        E_matrix = 0;
        K_inv = 0;
        local_g = 0;
        local_F = 0;
        F_native = 0;
        M_H_old = 0;
        V_vector = 0;
        Kinv_V = 0;
        b_local = 0;
        S_hat_local = 0;
        local_GHB = 0;

        // Get DoF indices
        flux_cell->get_dof_indices(flux_dof_indices);
        head_cell->get_dof_indices(head_dof_indices);
        trace_cell->get_dof_indices(trace_dof_indices);

        // Reinitialize FE values
        fe_values_flux.reinit(flux_cell);
        fe_values_head.reinit(head_cell);


    }



}

template <int dim>
void NPSAT_FLOW<dim>::identify_top_active_cells(
    std::vector<unsigned char> &recharge_receiver,
    std::vector<double> &receiver_recharge_area,
    std::vector<double> &receiver_effective_z_top) {

    const unsigned int n_local_cells = triangulation.n_locally_owned_active_cells();
    recharge_receiver.assign(n_local_cells, 0);
    receiver_recharge_area.assign(n_local_cells, 0.0);
    receiver_effective_z_top.assign(n_local_cells, std::numeric_limits<double>::quiet_NaN());
    std::unordered_set<std::uint64_t> current_recharge_receiver_gids;

    if (uo.sim_opt.confined) {
        unsigned int top_face = 5;
        for (auto trace_cell = dof_handler_trace.begin_active();
             trace_cell != dof_handler_trace.end();
             ++trace_cell)
        {
            if (!trace_cell->is_locally_owned())
                continue;

            if (!trace_cell->face(top_face)->at_boundary())
                continue;

            const unsigned int slot = trace_cell->user_index();
            AssertIndexRange(slot, recharge_receiver.size());

            recharge_receiver[slot] = 1;
            receiver_recharge_area[slot] += trace_cell->face(top_face)->measure();
        }

        previous_recharge_receiver_gids.clear();
        return;
    }

    const unsigned int n_head_dofs = fe_head.n_dofs_per_cell();
    std::vector<types::global_dof_index> head_dof_indices(n_head_dofs);

    using TraceCell = typename DoFHandler<dim>::active_cell_iterator;

    std::unordered_map<std::uint64_t, TraceCell> active_cell_by_gid;
    active_cell_by_gid.reserve(triangulation.n_active_cells());

    for (auto trace_cell = dof_handler_trace.begin_active();
         trace_cell != dof_handler_trace.end();
         ++trace_cell)
    {
        active_cell_by_gid.emplace(
            static_cast<std::uint64_t>(trace_cell->global_active_cell_index()),
            trace_cell);
    }

    std::vector<npsat_flow::RechargeRouteRequest> pending_requests;
    std::vector<std::vector<npsat_flow::RechargeRouteRequest>> send_requests(n_proc);

    const auto queue_cell = [&](const TraceCell &trace_cell, const double source_area)
    {
        AssertThrow(source_area >= 0.0,
                    ExcMessage("Recharge routing received a negative source area."));

        if (trace_cell->is_locally_owned())
        {
            pending_requests.push_back(
                npsat_flow::RechargeRouteRequest{
                    static_cast<std::uint64_t>(trace_cell->global_active_cell_index()),
                    source_area});
            return;
        }

        const unsigned int owner = trace_cell->subdomain_id();
        AssertThrow(owner < n_proc,
                    ExcMessage("Recharge routing reached a remote cell with invalid owner rank."));

        send_requests[owner].push_back(
            npsat_flow::RechargeRouteRequest{
                static_cast<std::uint64_t>(trace_cell->global_active_cell_index()),
                source_area});
    };

    const auto route_local_request = [&](const npsat_flow::RechargeRouteRequest &request)
    {
        const auto cell_it = active_cell_by_gid.find(request.cell_gid);
        AssertThrow(cell_it != active_cell_by_gid.end(),
                    ExcMessage("Recharge routing could not find active cell on owning rank."));

        const TraceCell &trace_cell = cell_it->second;
        AssertThrow(trace_cell->is_locally_owned(),
                    ExcMessage("Recharge routing request was delivered to a rank that does not own the cell."));

        const auto head_cell = trace_cell->as_dof_handler_iterator(dof_handler_head);
        head_cell->get_dof_indices(head_dof_indices);

        npsat_flow::CellNonlinearData cell_data;
        compute_cell_r_and_storage(cell_data, head_cell, head_dof_indices);

        constexpr bool verbose_recharge_routing = false;
        const double drying_saturated_fraction =
            std::max(0.0, uo.NLC.recharge_drying_saturated_fraction);
        const double wetting_saturated_fraction =
            std::max(drying_saturated_fraction,
                     uo.NLC.recharge_wetting_saturated_fraction);
        const double min_recharge_relative_k =
            std::max(0.0, uo.NLC.recharge_min_relative_k);
        const bool was_previous_receiver =
            previous_recharge_receiver_gids.find(request.cell_gid) !=
            previous_recharge_receiver_gids.end();
        const double selected_saturated_fraction =
            uo.NLC.use_recharge_hysteresis
            ? (was_previous_receiver ? drying_saturated_fraction
                                     : wetting_saturated_fraction)
            : drying_saturated_fraction;

        const double min_active_saturated_thickness =
            std::max(r_params.eps,
                     selected_saturated_fraction * cell_data.thickness);
        const bool enough_saturated_thickness =
            (cell_data.psi > min_active_saturated_thickness);
        const bool enough_relative_k =
            (cell_data.r > min_recharge_relative_k);
        const bool accepts_recharge =
            enough_saturated_thickness && enough_relative_k;

        if (verbose_recharge_routing &&
            (cell_data.is_partially_saturated || cell_data.is_fully_dry))
        {
            std::cout << std::setprecision(16)
                      << "Recharge routing dry/partial cell diagnostic on rank "
                      << my_rank << ":\n"
                      << "  cell_gid=" << request.cell_gid
                      << " active_index=" << trace_cell->active_cell_index()
                      << " slot=" << trace_cell->user_index()
                      << " center=" << trace_cell->center()
                      << " source_area=" << request.source_area << "\n"
                      << "  dry=" << cell_data.is_fully_dry
                      << " partial=" << cell_data.is_partially_saturated
                      << " saturated=" << cell_data.is_fully_saturated << "\n"
                      << "  psi=" << cell_data.psi
                      << " thickness=" << cell_data.thickness
                      << " min_psi=" << min_active_saturated_thickness
                      << " theta=" << (cell_data.psi / std::max(cell_data.thickness, 1e-12))
                      << " min_theta=" << selected_saturated_fraction
                      << " previous_receiver=" << was_previous_receiver << "\n"
                      << "  r=" << cell_data.r
                      << " min_r=" << min_recharge_relative_k
                      << " S_eff=" << cell_data.S_eff
                      << " accepts_recharge=" << accepts_recharge
                      << " thickness_ok=" << enough_saturated_thickness
                      << " relative_k_ok=" << enough_relative_k
                      << std::endl;
        }

        if (accepts_recharge)
        {
            const unsigned int slot = trace_cell->user_index();
            AssertIndexRange(slot, recharge_receiver.size());

            recharge_receiver[slot] = 1;
            receiver_recharge_area[slot] += request.source_area;
            receiver_effective_z_top[slot] = cell_data.h_e;
            current_recharge_receiver_gids.insert(request.cell_gid);
            return;
        }

        constexpr unsigned int bottom_face = 4;

        AssertThrow(!trace_cell->face(bottom_face)->at_boundary(),
                    ExcMessage("Recharge routing reached the bottom boundary before finding an active saturated cell."));

        const auto neighbor = trace_cell->neighbor(bottom_face);

        if (neighbor->is_active())
        {
            queue_cell(neighbor, request.source_area);
            return;
        }

        const unsigned int n_subfaces = trace_cell->face(bottom_face)->n_children();
        for (unsigned int subface = 0; subface < n_subfaces; ++subface)
        {
            const auto child = trace_cell->neighbor_child_on_subface(bottom_face, subface);
            AssertThrow(child->is_active(),
                        ExcMessage("Expected active child below refined bottom face."));

            const double child_area = child->face(5)->measure();
            queue_cell(child, child_area);
        }
    };

    constexpr unsigned int top_face = 5;

    for (auto trace_cell = dof_handler_trace.begin_active();
         trace_cell != dof_handler_trace.end();
         ++trace_cell)
    {
        if (!trace_cell->is_locally_owned())
            continue;

        if (!trace_cell->face(top_face)->at_boundary())
            continue;

        queue_cell(trace_cell, trace_cell->face(top_face)->measure());
    }

    while (true)
    {
        for (unsigned int r = 0; r < n_proc; ++r)
            send_requests[r].clear();

        for (std::size_t next = 0; next < pending_requests.size(); ++next)
            route_local_request(pending_requests[next]);

        pending_requests.clear();

        std::vector<int> send_counts(n_proc, 0), recv_counts(n_proc, 0);
        for (unsigned int r = 0; r < n_proc; ++r)
            send_counts[r] = static_cast<int>(send_requests[r].size());

        int ierr = MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                                recv_counts.data(), 1, MPI_INT,
                                mpi_communicator);
        AssertThrow(ierr == MPI_SUCCESS,
                    ExcMessage("MPI_Alltoall failed during recharge routing."));

        std::vector<int> send_displs(n_proc, 0), recv_displs(n_proc, 0);
        int send_total = 0;
        int recv_total = 0;
        for (unsigned int r = 0; r < n_proc; ++r)
        {
            send_displs[r] = send_total;
            send_total += send_counts[r];

            recv_displs[r] = recv_total;
            recv_total += recv_counts[r];
        }

        std::vector<npsat_flow::RechargeRouteRequest> send_flat;
        send_flat.reserve(static_cast<std::size_t>(send_total));
        for (unsigned int r = 0; r < n_proc; ++r)
            for (const auto &request : send_requests[r])
                send_flat.push_back(request);

        pending_requests.resize(static_cast<std::size_t>(recv_total));

        const int bytes_per_request = static_cast<int>(sizeof(npsat_flow::RechargeRouteRequest));
        std::vector<int> send_counts_b(n_proc, 0), recv_counts_b(n_proc, 0);
        std::vector<int> send_displs_b(n_proc, 0), recv_displs_b(n_proc, 0);
        for (unsigned int r = 0; r < n_proc; ++r)
        {
            send_counts_b[r] = send_counts[r] * bytes_per_request;
            recv_counts_b[r] = recv_counts[r] * bytes_per_request;
            send_displs_b[r] = send_displs[r] * bytes_per_request;
            recv_displs_b[r] = recv_displs[r] * bytes_per_request;
        }

        ierr = MPI_Alltoallv(reinterpret_cast<const char *>(send_flat.data()),
                             send_counts_b.data(),
                             send_displs_b.data(),
                             MPI_BYTE,
                             reinterpret_cast<char *>(pending_requests.data()),
                             recv_counts_b.data(),
                             recv_displs_b.data(),
                             MPI_BYTE,
                             mpi_communicator);
        AssertThrow(ierr == MPI_SUCCESS,
                    ExcMessage("MPI_Alltoallv failed during recharge routing."));

        const unsigned int local_pending =
            static_cast<unsigned int>(pending_requests.size());
        const unsigned int global_pending =
            Utilities::MPI::sum(local_pending, mpi_communicator);

        if (global_pending == 0)
            break;
    }

    previous_recharge_receiver_gids.swap(current_recharge_receiver_gids);




}

#endif //NPSAT_FLOW_ASSEMBLE_IMPL_H
