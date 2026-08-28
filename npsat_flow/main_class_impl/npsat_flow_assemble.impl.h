//
// Created by giorgk on 6/25/26.
//

#ifndef NPSAT_FLOW_ASSEMBLE_IMPL_H
#define NPSAT_FLOW_ASSEMBLE_IMPL_H

template <int dim>
void NPSAT_FLOW<dim>::assemble_system() {
    if (uo.verbose_level > 0)
    {
        pcout << "Assembling system for simulation counter "
              << time_tracking.simulation_step()
              << ", input data time step " << time_tracking.file_step();
        if (uo.sim_opt.steady_state)
            pcout << " (steady state)..." << std::endl;
        else
            pcout << " (duration " << time_tracking.duration() << ")..."
                  << std::endl;
    }
    double delta_time = time_tracking.duration();
    // flow_scale multiplies the spatial balance after condensation. It is dt
    // for backward-Euler transient flow and one for the unscaled steady
    // equations. The transient expressions below are otherwise unchanged.
    const double flow_scale = uo.sim_opt.steady_state ? 1.0 : delta_time;

    TimerOutput::Scope t(this->computing_timer, "assemble");

    typedef std::chrono::steady_clock AssemblyPhaseClock;
    const auto start_assembly_phase = [&](const std::string &phase_name)
    {
        if (uo.verbose_level > 0)
            pcout << "ASM phase: " << phase_name << " started" << std::endl;
        return AssemblyPhaseClock::now();
    };
    const auto finish_assembly_phase = [&](const std::string &phase_name,
                                           const AssemblyPhaseClock::time_point &phase_start)
    {
        if (uo.verbose_level > 0)
        {
            const double local_seconds =
                std::chrono::duration<double>(AssemblyPhaseClock::now() - phase_start).count();
            const double global_seconds =
                Utilities::MPI::max(local_seconds, mpi_communicator);
            pcout << "ASM phase: " << phase_name << " finished in "
                  << std::fixed << std::setprecision(2) << global_seconds
                  << " s" << std::defaultfloat << std::endl;
        }
    };

    AssemblyPhaseClock::time_point assembly_phase_start =
        start_assembly_phase("clearing matrices and right-hand sides");
    { // Make system matrices zero
        for (unsigned int i = 0; i < block_system_matrix.n_block_rows(); ++i)
            for (unsigned int j = 0; j < block_system_matrix.n_block_cols(); ++j)
                block_system_matrix.block(i, j) = 0;

        for (unsigned int b = 0; b < block_rhs_vector.n_blocks(); ++b)
            block_rhs_vector.block(b) = 0;
    }
    finish_assembly_phase("clearing matrices and right-hand sides",
                          assembly_phase_start);

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

    std::vector<std::vector<npsat_flow::Well10Entry>> send_10(n_proc); // for block(1,0)
    std::vector<std::vector<npsat_flow::Well11Entry>> send_11(n_proc); // for block(1,1)
    std::vector<std::vector<npsat_flow::WellRhsEntry>>  send_rhs1(n_proc);  // rhs block(1)
    std::vector<std::vector<npsat_flow::Trace01Entry>> send_01(n_proc); // for block(0,1)

    // Recharge values on faces (used only when the face has recharge BC)
    std::vector<double> recharge_values(n_face_q_points);
    std::vector<double> face_x;// These are used for the stream contribution
    std::vector<double> face_y;
    std::vector<double> stream_xc;
    std::vector<double> stream_yc;
    std::vector<double> stream_area;
    std::vector<double> stream_q;
    std::vector<double> ghb_head_values(n_face_q_points);
    std::vector<double> ghb_cond_values(n_face_q_points);

    // Global–local DOF index buffers
    std::vector<types::global_dof_index> flux_dof_indices(n_flux_dofs);
    std::vector<types::global_dof_index> head_dof_indices(n_head_dofs);
    std::vector<types::global_dof_index> trace_dof_indices(n_trace_dofs);

    const unsigned int n_local_cells = triangulation.n_locally_owned_active_cells();
    std::vector<unsigned char> recharge_receiver(n_local_cells, 0);
    std::vector<double> receiver_recharge_area(n_local_cells, 0.0);
    std::vector<double> receiver_effective_z_top(n_local_cells,std::numeric_limits<double>::quiet_NaN());
    if (uo.sim_opt.confined)
    {
        previous_recharge_receiver_gids.clear();
        if (uo.verbose_level > 0)
            pcout << "ASM phase: identifying recharge receiver cells skipped "
                     "(confined; recharge is applied directly to boundary face 5)"
                  << std::endl;
    }
    else
    {
        assembly_phase_start = start_assembly_phase("identifying recharge receiver cells");
        identify_top_active_cells(recharge_receiver,
                                  receiver_recharge_area,
                                  receiver_effective_z_top);
        finish_assembly_phase("identifying recharge receiver cells",
                              assembly_phase_start);
    }

    double local_recharge_total = 0.0;          // positive into the aquifer
    double local_stream_total = 0.0;            // positive into the aquifer
    double local_well_prescribed_total = 0.0;   // same sign convention as mnwells.pumping_rate()

    const unsigned int n_wells = static_cast<unsigned int>(mnwells.wells.size());
    const double wet_screen_length_tol = 1e-10;
    std::vector<double> local_well_cwc_eff_sum(n_wells, 0.0);
    std::vector<double> local_well_wet_screen_sum(n_wells, 0.0);
    std::vector<double> local_well_total_screen_sum(n_wells, 0.0);
    std::vector<double> local_well_bottom_head(
        n_wells, -std::numeric_limits<double>::max());


    // =====================================================
    // DEFINE LOCAL MATRICES ACCORDING TO THEORY
    // =====================================================
    // Transient:
    // [A   B^T   C^T] [Q]   [0]
    // [B   M/dt   0 ] [H] = [F + M*H_old/dt]
    // [C    0     0 ] [Λ]   [g]
    //
    // Steady:
    // [A   B^T   C^T] [Q]   [0]
    // [B    0     0 ] [H] = [F]
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

    const unsigned int progress_frequency =
        static_cast<unsigned int>(uo.assembly_progress_frequency);
    unsigned int next_progress_percent =
        progress_frequency == 0 ? 101u : progress_frequency;
    unsigned long long locally_assembled_cells = 0;
    const unsigned long long global_cell_count =
        static_cast<unsigned long long>(triangulation.n_global_active_cells());
    assembly_phase_start = start_assembly_phase("assembling local cells");
    const AssemblyPhaseClock::time_point assembly_cell_loop_start =
        AssemblyPhaseClock::now();

    const auto report_assembly_progress = [&]()
    {
        while (next_progress_percent <= 100u)
        {
            const unsigned long long local_threshold =
                (static_cast<unsigned long long>(next_progress_percent) * n_local_cells + 99u) / 100u;
            if (locally_assembled_cells < local_threshold)
                break;

            unsigned long long globally_assembled_cells = 0;
            const int ierr = MPI_Allreduce(&locally_assembled_cells,
                                           &globally_assembled_cells,
                                           1,
                                           MPI_UNSIGNED_LONG_LONG,
                                           MPI_SUM,
                                           mpi_communicator);
            AssertThrow(ierr == MPI_SUCCESS,
                        ExcMessage("MPI_Allreduce failed while reporting assembly progress."));

            const double local_elapsed_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              assembly_cell_loop_start).count();
            const double elapsed_seconds =
                Utilities::MPI::max(local_elapsed_seconds, mpi_communicator);
            const double completed_fraction =
                global_cell_count > 0
                    ? static_cast<double>(globally_assembled_cells) /
                          static_cast<double>(global_cell_count)
                    : 1.0;
            const double eta_seconds =
                completed_fraction > 0.0
                    ? elapsed_seconds * (1.0 - completed_fraction) / completed_fraction
                    : 0.0;

            pcout << "Assembly progress: " << globally_assembled_cells
                  << " / " << global_cell_count << " cells ("
                  << std::fixed << std::setprecision(1)
                  << 100.0 * completed_fraction << "%), elapsed "
                  << std::setprecision(1) << elapsed_seconds << " s, ETA "
                  << std::max(0.0, eta_seconds) << " s"
                  << std::defaultfloat << std::endl;

            if (next_progress_percent == 100u)
                next_progress_percent = 101u;
            else
                next_progress_percent =
                    std::min(100u, next_progress_percent + progress_frequency);
        }
    };

    // Ranks without owned cells still participate in every progress collective.
    report_assembly_progress();

    for (auto trace_cell = dof_handler_trace.begin_active(); trace_cell != dof_handler_trace.end(); ++trace_cell) {
        //pcout << "Cell index: " << cell_index << std::endl;
        if (!trace_cell->is_locally_owned())
            continue;

        const unsigned int slot = trace_cell->user_index();
        Assert(slot != numbers::invalid_unsigned_int, ExcInternalError());
        local_element_data_rt_0dg0.assert_valid_slot(slot);

        // Use the same physical cell to get indices from each handler:
        typename DoFHandler<dim>::active_cell_iterator flux_cell(&triangulation, trace_cell->level(), trace_cell->index(), &dof_handler_flux);
        typename DoFHandler<dim>::active_cell_iterator head_cell(&triangulation, trace_cell->level(), trace_cell->index(), &dof_handler_head);
        //auto flux_cell = trace_cell->as_dof_handler_iterator(dof_handler_flux);
        //auto head_cell = trace_cell->as_dof_handler_iterator(dof_handler_head);

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

        npsat_flow::CellNonlinearData nl_cell_data;
        compute_cell_r_and_storage(nl_cell_data, head_cell, head_dof_indices);
        const double routed_receiver_effective_z_top =
            (!uo.sim_opt.confined && recharge_receiver[slot] != 0)
                ? receiver_effective_z_top[slot]
                : std::numeric_limits<double>::quiet_NaN();
        const double selected_effective_z_top = effective_top_for_cell(nl_cell_data, routed_receiver_effective_z_top);
        if (std::isfinite(selected_effective_z_top))
            compute_cell_r_and_storage(nl_cell_data, head_cell, head_dof_indices, selected_effective_z_top);

        // Get Hⁿ for this cell
        Vector<double> H_old_local(n_head_dofs);
        for (unsigned int k = 0; k < n_head_dofs; ++k) {
            H_old_local(k) = h_old[head_dof_indices[k]];
        }
        print_vector(H_old_local, "H_old_local");

        // There is no skipping
        // =====================================================
        // ASSEMBLE LOCAL MATRICES
        // =====================================================
        double cell_volume = 0;
        for (unsigned int q = 0; q < n_q_points; ++q) {
            const Point<dim> &p = fe_values_flux.quadrature_point(q);
            const double JxW = fe_values_flux.JxW(q);
            cell_volume += JxW;
            const Tensor<2, dim> Kmat_inv = hgeo_prop.conductivity_inverse(p);
            // K_eff = r * K  =>  K_eff^{-1} = (1/r) * K^{-1}
            const double r = nl_cell_data.r;
            const double inv_r = 1.0 / r;
            Tensor<2, dim> K_eff_inv = inv_r * Kmat_inv;

            const double S_eff = nl_cell_data.S_eff;

            // --------------------------------------------------
            // Matrix A: ∫ K⁻¹ φ_i · φ_j dΩ  (n_q × n_q)
            // --------------------------------------------------
            for (unsigned int i = 0; i < n_flux_dofs; ++i)
            {
                const Tensor<1, dim> phi_i = fe_values_flux[flux].value(i, q);
                for (unsigned int j = 0; j < n_flux_dofs; ++j)
                {
                    const Tensor<1, dim> phi_j = fe_values_flux[flux].value(j, q);
                    local_A(i, j) += (K_eff_inv * phi_i * phi_j) * JxW;
                }
            }

            // --------------------------------------------------
            // B: -∫ ψ_k (∇·φ_i) dΩ  (n_h × n_q)
            // --------------------------------------------------
            // Note: k = head dof index (row), i = flux dof index (col)
            for (unsigned int k = 0; k < n_head_dofs; ++k)
            {
                const double psi_k = fe_values_head.shape_value(k, q);
                for (unsigned int i = 0; i < n_flux_dofs; ++i)
                {
                    const double div_phi_i = fe_values_flux[flux].divergence(i, q);
                    //local_B(k, i) += -psi_k * div_phi_i * JxW;
                    // See other repo for justification for reversing the sign
                    local_B(k, i) += -psi_k * div_phi_i * JxW;
                }
            }

            // --------------------------------------------------
            // Matrix M: ∫ S_s ψ_k ψ_l dΩ  (n_h × n_h)
            // --------------------------------------------------
            if (!uo.sim_opt.steady_state)
                for (unsigned int k = 0; k < n_head_dofs; ++k)
                {
                    const double psi_k = fe_values_head.shape_value(k, q);
                    for (unsigned int l = 0; l < n_head_dofs; ++l)
                    {
                        const double psi_l = fe_values_head.shape_value(l, q);
                        local_M(k, l) += S_eff * psi_k * psi_l * JxW;
                    }
                }
        }// End of quadrature point loop
        //pcout << "Cell " << cell_index << " volume: " << cell_volume << " m³" << std::endl;
        print_matrix(local_A, "local_A");
        print_matrix(local_B, "local_B");
        print_matrix(local_M, "local_M");

        {// If there are any volumetric sources or sinks they should be assembled here
            // For example Wells or stream as volumes

        }

        // =====================================================
        // ASSEMBLE MATRIX C AND BOUNDARY CONDITIONS
        // =====================================================
        for (unsigned int i_face = 0; i_face < GeometryInfo<dim>::faces_per_cell; ++i_face) {
            // Initialize face values
            fe_face_values_flux.reinit(flux_cell, i_face);
            fe_face_values_trace.reinit(trace_cell, i_face);

            // =====================================================
            // PART 1: ASSEMBLE C MATRIX (ALWAYS THE SAME)
            // =====================================================
            for (unsigned int q = 0; q < n_face_q_points; ++q) {
                const double JxW = fe_face_values_trace.JxW(q);
                const Tensor<1, dim> normal = fe_face_values_flux.normal_vector(q);

                for (unsigned int l_face = 0; l_face < fe_trace.dofs_per_face; ++l_face) {
                    const unsigned int l_cell = fe_trace.face_to_cell_index(l_face, i_face);

                    // // Safer: use face dof index for face shape value // TODO Check which one is correct
                    // const double chi_l = fe_face_values_trace.shape_value(l_face, q);
                    const double chi_l = fe_face_values_trace.shape_value(l_cell, q);

                    for (unsigned int i = 0; i < n_flux_dofs; ++i)
                    {
                        const Tensor<1, dim> phi_i = fe_face_values_flux[flux].value(i, q);
                        const double phi_i_dot_n = phi_i * normal;
                        local_C(l_cell, i) += chi_l * phi_i_dot_n * JxW;
                    }
                }
            }// End of quadrature points loop

            // =====================================================
            // PART 2: ADD RHS CONTRIBUTIONS
            // =====================================================
            if (should_receive_gw_recharge(recharge_receiver, flux_cell, i_face)) {
                const unsigned int slot = trace_cell->user_index();
                AssertIndexRange(slot, receiver_recharge_area.size());

                const double top_face_area = flux_cell->face(i_face)->measure();
                AssertThrow(top_face_area > 0.0, ExcMessage("Recharge receiver top face has zero measure."));
                // In confined simulations every top boundary face receives its
                // own recharge directly. Area accumulation/scaling is needed
                // only when unconfined recharge has been routed between cells.
                const double recharge_area_scale =
                    uo.sim_opt.confined
                        ? 1.0
                        : receiver_recharge_area[slot] / top_face_area;

                // Assuming gw_recharge is in m/day (flux per unit area)
                // This is the typical unit for groundwater recharge
                gw_recharge.value_list(fe_face_values_trace.get_quadrature_points(), recharge_values);

                for (unsigned int q = 0; q < n_face_q_points; ++q) {
                    // Face measure (same for flux and trace; you use either)
                    const double JxW_tr = fe_face_values_trace.JxW(q);

                    // Recharge rate in m/day (flux per unit area)
                    const double recharge = recharge_values[q];     // [L/T], positive downward into aquifer
                    const double gN       = -recharge * recharge_area_scale; // q·n outward: negative for inflow
                    local_recharge_total += recharge * recharge_area_scale * JxW_tr;

                    // ------------------------------------------------------------
                    // (A) Trace RHS: local_g(l) += ∫ (q·n prescribed) * χ_l dΓ
                    //
                    // Your global trace equation is of the form:
                    //   C Q = g
                    // so local_g must represent ∫ gN χ dΓ in consistent sign convention.
                    // ------------------------------------------------------------
                    for (unsigned int l_face = 0; l_face < fe_trace.dofs_per_face; ++l_face)
                    {
                        const unsigned int l_cell = fe_trace.face_to_cell_index(l_face, i_face);
                        //const double chi_l = fe_face_values_trace.shape_value(l_face, q); //TODO maybe this is correct
                        const double chi_l = fe_face_values_trace.shape_value(l_cell, q);
                        // g_l += ∫ gN * χ_l dΓ
                        local_g(l_cell) += gN * chi_l * JxW_tr;
                    }
                }

                npsat_flow::get_face_xy_polygon<dim>(trace_cell, i_face, face_x, face_y);
                if (streams.find_stream_recharge_in_polygon(stream_xc, stream_yc, stream_area, stream_q, face_x, face_y)) {
                    AssertThrow(stream_xc.size() == stream_yc.size() && stream_xc.size() == stream_area.size() &&
                        stream_xc.size() == stream_q.size(), ExcMessage("Inconsistent stream recharge intersection data."));

                    for (unsigned int k = 0; k < stream_q.size(); ++k) {
                        (void)stream_area[k];
                        Point<dim - 1> unit_point;
                        if (!npsat_flow::map_xy_to_unit_face<dim>(trace_cell, i_face, stream_xc[k], stream_yc[k], unit_point))
                            continue;

                        const Quadrature<dim - 1> stream_quadrature(unit_point);
                        FEFaceValues<dim> fe_face_stream_values(fe_trace, stream_quadrature, update_values);
                        fe_face_stream_values.reinit(trace_cell, i_face);

                        const double stream_flow = stream_q[k]; // [L^3/T], positive into aquifer
                        const double gN_integral = -stream_flow;
                        local_stream_total += stream_flow;

                        for (unsigned int l_face = 0; l_face < fe_trace.dofs_per_face; ++l_face)
                        {
                            const unsigned int l_cell = fe_trace.face_to_cell_index(l_face, i_face);
                            const double chi_l = fe_face_stream_values.shape_value(l_cell, 0);
                            local_g(l_cell) += gN_integral * chi_l;
                        }
                    }
                }
            }

            if (flux_cell->face(i_face)->at_boundary()) {
                const types::boundary_id boundary_id = flux_cell->face(i_face)->boundary_id();

                // General Head Boundary conditions
                const auto ghb_it = ghb_boundary_map.find(boundary_id);
                if (ghb_it != ghb_boundary_map.end()) {
                    const auto &ghb_functions = ghb_it->second;
                    AssertThrow(ghb_functions.head != nullptr, ExcMessage("GHB head function is null."));
                    AssertThrow(ghb_functions.conductance != nullptr, ExcMessage("GHB conductance function is null."));

                    ghb_functions.head->value_list( fe_face_values_trace.get_quadrature_points(), ghb_head_values);
                    ghb_functions.conductance->value_list(fe_face_values_trace.get_quadrature_points(), ghb_cond_values);
                    for (unsigned int q = 0; q < n_face_q_points; ++q) {
                        const double JxW = fe_face_values_trace.JxW(q);
                        const double h_b = ghb_head_values[q];
                        const double c   = ghb_cond_values[q];
                        AssertThrow(c >= 0.0, ExcMessage("GHB conductance must be nonnegative."));
                        for (unsigned int i_face_dof  = 0; i_face_dof < fe_trace.dofs_per_face; ++i_face_dof ) {
                            const unsigned int i_cell = fe_trace.face_to_cell_index(i_face_dof, i_face);
                            const double chi_i = fe_face_values_trace.shape_value(i_cell, q);
                            local_g(i_cell) += -c * h_b * chi_i * JxW;
                            for (unsigned int j_face_dof  = 0; j_face_dof  < fe_trace.dofs_per_face; ++j_face_dof ) {
                                const unsigned int j_cell = fe_trace.face_to_cell_index(j_face_dof , i_face);
                                const double chi_j = fe_face_values_trace.shape_value(j_cell, q);
                                local_GHB(i_cell, j_cell) += c * chi_i * chi_j * JxW;
                            }
                        }
                    }
                }
                // For other boundary types:
                // - Dirichlet: Handled via constraints (no RHS here)
                // - No-flow: Neumann with g_N = 0 (no RHS contribution needed)
                // - Other Neumann: Add similar to recharge
            }
        }// End of face loop
        print_vector(local_g, "local_g");
        print_vector(F_native, "F_native");
        print_matrix(local_C, "local_C");

        // =====================================================
        // STATIC CONDENSATION FOR HYBRIDIZED SYSTEM
        // =====================================================
        // 1. Compute A_inv
        // 2. Compute D = B * A⁻¹ * B^T
        // 3. Compute K = M - Δt * D
        // 4. Compute E = B * A⁻¹ * C^T
        // 5. Compute S = C * A⁻¹ * C^T
        // 6. Global matrix: Ŝ = Δt * (E^T * K⁻¹ * E) + S
        // 7. Global RHS: b = -g - E^T * K⁻¹ * [M * Hⁿ + Δt * F]

        // =====================================================
        // COMPUTE LOCAL INVERSE OF A (or factorization)
        // =====================================================
        // We need A⁻¹ multiple times, so compute it once
        // Create copy of A for inversion
        //FullMatrix<double> A_inv(n_flux_dofs, n_flux_dofs);
        A_inv = local_A;
        try
        {
            A_inv.gauss_jordan();
        }
        catch (...)
        {
            // A might be singular (e.g., for very low permeability)
            // Add stabilization: A_stab = A + α I
            const double alpha = 1e-12;
            for (unsigned int i = 0; i < n_flux_dofs; ++i)
                A_inv(i, i) += alpha;
            A_inv.gauss_jordan();
        }
        print_matrix(A_inv, "A_inv");

        // =====================================================
        // COMPUTE D = B * A⁻¹ * B^T  (Schur complement for H) (n_h × n_h matrix)
        // =====================================================
        FullMatrix<double> D_matrix(n_head_dofs, n_head_dofs);
        D_matrix.triple_product(A_inv, local_B, local_B, false, true);
        print_matrix(D_matrix, "D_matrix");

        // =====================================================
        // COMPUTE K: transient M + dt*D; steady D
        // =====================================================
        FullMatrix<double> K_matrix(n_head_dofs, n_head_dofs);
        if (uo.sim_opt.steady_state)
            K_matrix.add(1.0, D_matrix);
        else
            K_matrix.add(1.0, local_M, delta_time, D_matrix);
        print_matrix(K_matrix, "K_matrix");

        // Check if K is invertible (should be SPD for dt small enough)
        // If not, we need stabilization
        if (K_matrix.trace() < 1e-14)
        {
            for (unsigned int i = 0; i < n_head_dofs; ++i)
                K_matrix(i, i) += 1e-10;
        }

        // Invert K for later use
        //FullMatrix<double> K_inv(n_head_dofs, n_head_dofs);
        K_inv = K_matrix;
        K_inv.gauss_jordan();
        print_matrix(K_inv, "K_inv");

        // =====================================================
        // COMPUTE E = B * A⁻¹ * C^T  (n_h × n_λ matrix)
        // =====================================================
        //FullMatrix<double> E_matrix(n_head_dofs, n_trace_dofs);
        E_matrix.triple_product(A_inv, local_B, local_C, false, true);
        print_matrix(E_matrix, "E_matrix");

        // =====================================================
        // COMPUTE S = C * A⁻¹ * C^T (n_λ × n_λ matrix)
        // =====================================================
        FullMatrix<double> S_matrix(n_trace_dofs, n_trace_dofs);
        S_matrix.triple_product(A_inv, local_C, local_C, false, true);
        print_matrix(S_matrix, "S_matrix");

        // =====================================================
        // COMPUTE LOCAL CONTRIBUTION TO GLOBAL MATRIX
        // =====================================================
        // Global trace matrix:
        // transient S - dt*E^T*K^-1*E; steady S - E^T*D^-1*E.
        //FullMatrix<double> S_hat_local(n_trace_dofs, n_trace_dofs);
        S_hat_local = S_matrix;
        S_hat_local.triple_product(K_inv, E_matrix, E_matrix, true, false, -flow_scale);
        S_hat_local.add(1.0, local_GHB);
        print_matrix(S_hat_local, "S_hat_local");

        // =====================================================
        // COMPUTE LOCAL RHS CONTRIBUTION
        // =====================================================
        // Compute M * Hⁿ
        //Vector<double> M_H_old(n_head_dofs);
        if (!uo.sim_opt.steady_state)
            local_M.vmult(M_H_old, H_old_local);
        print_vector(M_H_old, "M_H_old");

        // Compute V: transient M*H_old + dt*F; steady F.
        //Vector<double> V_vector(n_head_dofs);
        if (uo.sim_opt.steady_state)
            V_vector = local_F;
        else
        {
            V_vector = M_H_old;
            V_vector.add(delta_time, local_F);
        }
        print_vector(V_vector, "V_vector");

        // Compute K⁻¹ * V
        //Vector<double> Kinv_V(n_head_dofs);
        K_inv.vmult(Kinv_V, V_vector);
        print_vector(Kinv_V, "Kinv_V");

        // Compute E^T * (K⁻¹ * V)
        Vector<double> ET_Kinv_V(n_trace_dofs);
        E_matrix.Tvmult(ET_Kinv_V, Kinv_V);
        print_vector(ET_Kinv_V, "ET_Kinv_V");

        // Compute b_local = -g - E^T * K⁻¹ * [M * Hⁿ + Δt * F]
        //Vector<double> b_local(n_trace_dofs);
        b_local = local_g;
        b_local *= -1.0;  // -g
        b_local -= ET_Kinv_V;  // - E^T * K⁻¹ * V
        print_vector(b_local, "b_local");


        // =====================================================
        // ASSEMBLE Multi Node Wells (MPI-correct, distributed well rows)
        // =====================================================
        {
            auto it = std::lower_bound(local_cell_well_map.begin(), local_cell_well_map.end(),
                    trace_cell->active_cell_index(),
                    [](const std::pair<unsigned int, std::vector<npsat_flow::CellWellLink>>& pair,
                        unsigned int val)
                    {
                        return pair.first < val;
                    });

            if (it != local_cell_well_map.end() && it->first == trace_cell->active_cell_index()) {
                // This cell has one or more wells!
                // =====================================================
                // COMPUTE ae: transient dt*K^-1*E; steady D^-1*E.
                // =====================================================
                K_inv.mmult(ae_matrix, E_matrix);
                ae_matrix *= flow_scale;
                local_element_data_rt_0dg0.set_ae(slot, ae_matrix);
                print_matrix(ae_matrix, "ae_matrix");

                // Loop through the wells of this cell
                for (const auto &link : it->second) {
                    // Access well-common data
                    const npsat_flow::well_t well_id = static_cast<npsat_flow::well_t>(link.well_global_index);
                    // Access cell-unique data
                    double wet_screen_length = 0.0;
                    const double cwc = npsat_flow::effective_well_link_conductance(link, nl_cell_data.h_e, uo.sim_opt.confined, &wet_screen_length);
                    AssertThrow(well_id < mnwells.wells.size(), ExcMessage("well_id out of range"));
                    local_well_cwc_eff_sum[well_id] += cwc;
                    local_well_wet_screen_sum[well_id] += wet_screen_length;
                    local_well_total_screen_sum[well_id] += link.sl;
                    const double bottom_tolerance =
                        1.0e-8 * (1.0 + std::abs(mnwells.wells[well_id].bottom));
                    if (std::abs(link.w_zbot - mnwells.wells[well_id].bottom) <=
                        bottom_tolerance)
                        local_well_bottom_head[well_id] =
                            std::max(local_well_bottom_head[well_id], nl_cell_data.h_e);

                    // Ownership of this well dof
                    const unsigned int well_owner = well_owner_rank[well_id];
                    // ================================================================
                    // (A) Correct local Schur matrix and local rhs (trace-side physics)
                    //     S_hat_local and b_local are LOCAL (cell) objects.
                    // ================================================================
                    for (unsigned int i = 0; i < trace_dof_indices.size(); ++i)
                    {
                        const double a_i = ae_matrix(0,i);

                        // NOTE: this assumes b_local is indexed by local trace dof order
                        // and corresponds to trace_dof_indices order
                        b_local(i) += cwc * a_i * Kinv_V(0);

                        for (unsigned int j = 0; j < trace_dof_indices.size(); ++j)
                        {
                            const double a_j = ae_matrix(0,j);
                            S_hat_local.add(i,j,cwc * a_i * a_j);
                        }
                    }

                    // ================================================================
                    // (B) Coupling blocks:
                    //     (0,1): trace-row, well-column  -> assemble immediately here
                    //     (1,0): well-row, trace-column  -> send to well owner
                    // ================================================================
                    // --------------------------------------------------
                    //    R coupling: trace-to-well / well-to-trace
                    //    block(0,1) = Ce
                    //    block(1,0) = Ce
                    // --------------------------------------------------
                    for (unsigned int i = 0; i < n_trace_dofs; ++i) {
                        const npsat_flow::dof_t trace_i = trace_dof_indices[i];
                        const double Re_contribution = cwc * ae_matrix(0, i);

                        if (!lambda_constraints.is_constrained(trace_i)) { // If the trace_dof its not constrained
                            // ---- (0,1): row is trace dof (owned here), col is well dof (may be remote)
                            // This is allowed if the sparsity pattern includes the entry.
                            if (lambda_locally_owned_dofs.is_element(trace_i)) {
                                block_system_matrix.block(0,1).add(trace_i, well_id, Re_contribution);
                            }
                            else {
                                // row not owned here -> send to row owner
                                const unsigned int row_owner = lambda_ownership.get_owner_cached(trace_i);
                                send_01[row_owner].push_back(npsat_flow::Trace01Entry{trace_i, well_id, Re_contribution});
                            }

                            if (well_owner == my_rank) {
                                block_system_matrix.block(1,0).add(well_id, trace_i, Re_contribution);
                            }
                            else {
                                //block_system_matrix.block(1,0).add(well_dof_index, trace_i,  Re_contribution);
                                // ---- (1,0): send to well owner (row=well_id)
                                send_10[well_owner].push_back(npsat_flow::Well10Entry{well_id, trace_i, Re_contribution});
                            }
                        }
                        else {// If the dof is constrained
                            // Distribute constrained trace dof -> masters
                            const auto *line = lambda_constraints.get_constraint_entries(trace_i);

                            if (line != nullptr) {
                                for (const auto &entry : *line) {
                                    const npsat_flow::dof_t master_dof = entry.first;
                                    const double weight = entry.second;
                                    const double v = weight * Re_contribution;

                                    // (0,1) assembled on trace side, only if I own the master row
                                    if (lambda_locally_owned_dofs.is_element(master_dof)) {
                                        block_system_matrix.block(0,1).add(master_dof, well_id, v);
                                    }
                                    else {
                                        // I do not own that master row -> route to the owner rank of 'master'
                                        const unsigned int master_owner = lambda_ownership.get_owner_cached(master_dof);
                                        send_01[master_owner].push_back(npsat_flow::Trace01Entry{master_dof, well_id, v});
                                    }

                                    if (well_owner == my_rank) {
                                        block_system_matrix.block(1,0).add(well_id, master_dof, v);
                                    }
                                    else {
                                        // (1,0) must go to well owner
                                        send_10[well_owner].push_back(npsat_flow::Well10Entry{well_id, master_dof, v});
                                    }
                                }
                            }

                            // Inhomogeneous constraint: ti = sum(weight*master) + b
                            const double inhom = lambda_constraints.get_inhomogeneity(trace_i);
                            if (std::abs(inhom) > 0.0)
                            {
                                const double rhs_val = -Re_contribution * inhom;
                                // Move val * ti(b) to RHS of well equation (and (optionally) trace eq if needed)
                                if (well_owner == my_rank) {
                                    block_rhs_vector.block(1)(well_id) += rhs_val;
                                }
                                else {
                                    send_rhs1[well_owner].push_back(npsat_flow::WellRhsEntry{well_id, rhs_val});
                                }
                            }
                        }
                    }

                    // ================================================================
                    // (C) Well-to-well block and well RHS contributions:
                    //     (1,1) and RHS(1) are well-row objects -> send to well owner
                    // ================================================================
                    //   rhs1(well) += cwc * Kinv_V(0)
                    if (well_owner == my_rank) {
                        block_system_matrix.block(1,1).add(well_id, well_id, cwc);
                        block_rhs_vector.block(1)(well_id) += cwc * Kinv_V(0); //H_old_local(0);
                    }
                    else {
                        send_11[well_owner].push_back(npsat_flow::Well11Entry{well_id, cwc});
                        send_rhs1[well_owner].push_back(npsat_flow::WellRhsEntry{well_id, cwc * Kinv_V(0)});
                    }
                }
            }
        }// End MNW cell-local block

        // =====================================================
        // ASSEMBLE INTO GLOBAL SYSTEM USING CONSTRAINTS
        // =====================================================
        // Use constraints to distribute local to global
        lambda_constraints.distribute_local_to_global(S_hat_local,
                                                     b_local,
                                                     trace_dof_indices,
                                                     block_system_matrix.block(0,0),
                                                     block_rhs_vector.block(0));

        // =====================================================
        // STORE LOCAL DATA FOR POST-PROCESSING
        // =====================================================
        // After solving for Λ, we need to recover H and Q:
        // Transient: H = K^-1[M*H_old + dt*F - dt*E*Lambda].
        // Steady:    H = D^-1[F - E*Lambda].
        // Both:      Q = -A^-1[B^T*H + C^T*Lambda].

        {
            local_element_data_rt_0dg0.set_E(slot,E_matrix);
            local_element_data_rt_0dg0.set_kinv(slot,K_inv(0,0));
            local_element_data_rt_0dg0.set_V(slot,V_vector[0]);
            local_element_data_rt_0dg0.set_A_inv(slot,A_inv);
            local_element_data_rt_0dg0.set_B(slot,local_B);
            local_element_data_rt_0dg0.set_C(slot,local_C);
            local_element_data_rt_0dg0.set_kinv_v(slot, Kinv_V[0]);
            const double storage_mass = local_M(0,0);
            AssertThrow(std::isfinite(storage_mass) && storage_mass >= 0.0,
                        ExcMessage("Cell storage mass must be finite and nonnegative."));
            local_element_data_rt_0dg0.set_M00(slot, storage_mass);
        }

        ++locally_assembled_cells;
        report_assembly_progress();
    }// End of active cells loop
    finish_assembly_phase("assembling local cells", assembly_phase_start);

    // ----------------------------------------------------------------------------
    // MPI exchange for values
    // ----------------------------------------------------------------------------
    assembly_phase_start = start_assembly_phase("exchanging MNW coupling data");
    std::vector<npsat_flow::Well10Entry>  recv_10;
    std::vector<npsat_flow::Well11Entry>  recv_11;
    std::vector<npsat_flow::WellRhsEntry> recv_rhs1;
    std::vector<npsat_flow::Trace01Entry> recv_01;

    std::vector<int> recvcounts_10, recvcounts_11, recvcounts_rhs1, recvcounts_01;

    // --- Build send stats BEFORE exchange (counts are from send buffers) ---
    auto st10  = npsat_flow::compute_send_stats(send_10,   mpi_communicator, "send_10  ");
    auto st11  = npsat_flow::compute_send_stats(send_11,   mpi_communicator, "send_11  ");
    auto strhs = npsat_flow::compute_send_stats(send_rhs1, mpi_communicator, "send_rhs1");
    auto st01  = npsat_flow::compute_send_stats(send_01,   mpi_communicator, "send_01  ");

    // --- Exchange values AND get per-source recvcounts ---
    npsat_flow::send_receive_mnw_well_row_data(send_10, send_11, send_rhs1,recv_10, recv_11, recv_rhs1,
        mpi_communicator,&recvcounts_10, &recvcounts_11, &recvcounts_rhs1);
    npsat_flow::send_receive_trace01_data(send_01, recv_01, mpi_communicator, &recvcounts_01);

    // --- Fill receive-side stats from recvcounts vectors ---
    npsat_flow::fill_receive_stats_from_recvcounts(st10,  recvcounts_10);
    npsat_flow::fill_receive_stats_from_recvcounts(st11,  recvcounts_11);
    npsat_flow::fill_receive_stats_from_recvcounts(strhs, recvcounts_rhs1);
    npsat_flow::fill_receive_stats_from_recvcounts(st01,  recvcounts_01);
    MPI_Barrier(mpi_communicator);

    // --- Print send + receive stats (ordered by rank) ---
    if (uo.verbose_level > 1)
    {
        npsat_flow::print_alltoall_stats_compact_aligned(st10,  mpi_communicator);
        npsat_flow::print_alltoall_stats_compact_aligned(st11,  mpi_communicator);
        npsat_flow::print_alltoall_stats_compact_aligned(strhs, mpi_communicator);
        npsat_flow::print_alltoall_stats_compact_aligned(st01,  mpi_communicator);
    }
    MPI_Barrier(mpi_communicator);

    // --- Verify global consistency of each exchange ---
    if (uo.verbose_level > 1)
    {
        npsat_flow::check_alltoall_consistency(st10,  mpi_communicator);
        npsat_flow::check_alltoall_consistency(st11,  mpi_communicator);
        npsat_flow::check_alltoall_consistency(strhs, mpi_communicator);
        npsat_flow::check_alltoall_consistency(st01,  mpi_communicator);
        MPI_Barrier(mpi_communicator);
    }

    // Apply on this rank (I own these well rows)
    for (const auto &e : recv_10)
        if (well_locally_owned_dofs.is_element(e.well))
            block_system_matrix.block(1,0).add(e.well, e.col, e.val);

    for (const auto &e : recv_11)
        if (well_locally_owned_dofs.is_element(e.well))
            block_system_matrix.block(1,1).add(e.well, e.well, e.val);

    for (const auto &e : recv_rhs1)
        if (well_locally_owned_dofs.is_element(e.well))
            block_rhs_vector.block(1)(e.well) += e.val;

    // Apply: only owned trace rows on this rank
    for (const auto &e : recv_01)
    {
        if (lambda_locally_owned_dofs.is_element(e.row))
            block_system_matrix.block(0,1).add(e.row, e.col, e.val);
    }
    finish_assembly_phase("exchanging MNW coupling data", assembly_phase_start);

    assembly_phase_start = start_assembly_phase("reducing global well data");
    std::vector<double> global_well_cwc_eff_sum(n_wells, 0.0);
    std::vector<double> global_well_wet_screen_sum(n_wells, 0.0);
    std::vector<double> global_well_total_screen_sum(n_wells, 0.0);
    std::vector<double> global_well_bottom_head(
        n_wells, -std::numeric_limits<double>::max());
    if (n_wells > 0) {
        MPI_Allreduce(local_well_cwc_eff_sum.data(),
                      global_well_cwc_eff_sum.data(),
                      static_cast<int>(n_wells),
                      MPI_DOUBLE,
                      MPI_SUM,
                      mpi_communicator);
        MPI_Allreduce(local_well_wet_screen_sum.data(),
                      global_well_wet_screen_sum.data(),
                      static_cast<int>(n_wells),
                      MPI_DOUBLE,
                      MPI_SUM,
                      mpi_communicator);
        MPI_Allreduce(local_well_total_screen_sum.data(),
                      global_well_total_screen_sum.data(),
                      static_cast<int>(n_wells),
                      MPI_DOUBLE,
                      MPI_SUM,
                      mpi_communicator);
        MPI_Allreduce(local_well_bottom_head.data(),
                      global_well_bottom_head.data(),
                      static_cast<int>(n_wells),
                      MPI_DOUBLE,
                      MPI_MAX,
                      mpi_communicator);
    }
    finish_assembly_phase("reducing global well data", assembly_phase_start);

    // --------------------------------------------------
    //   Well RHS: Qtot (prescribed pumping) add only once
    // Add prescribed pumping Q ONLY ON THE WELL OWNER (avoid n_proc overcount)
    // --------------------------------------------------
    unsigned int local_dry_well_count = 0;
    double local_dry_well_requested_total = 0.0;
    double local_requested_pumping_magnitude = 0.0;
    double local_pumping_loss_magnitude = 0.0;
    std::vector<std::string> local_dry_well_rows;

    // Validate and cache every pumping-row lookup on every rank. Previously,
    // an invalid q_row was queried only by the owning rank; that rank could
    // throw while all other ranks waited indefinitely in the budget reduction.
    assembly_phase_start = start_assembly_phase("validating well pumping rates");
    std::vector<double> well_pumping_rates(n_wells, 0.0);
    for (std::size_t well_position = 0;
         well_position < mnwells.wells.size();
         ++well_position)
    {
        const npsat_flow::MNWell &well = mnwells.wells[well_position];
        AssertThrow(well.global_index < n_wells,
                    ExcMessage("MNWell.global_index is outside the global well range."));
        well_pumping_rates[well.global_index] = mnwells.pumping_rate(well.q_row);
    }
    finish_assembly_phase("validating well pumping rates", assembly_phase_start);

    assembly_phase_start = start_assembly_phase("assembling owned-well right-hand sides");

    for (const auto &well : mnwells.wells)
    {
        const unsigned int w_id = well.global_index;
        if (well_owner_rank[w_id] == static_cast<unsigned int>(my_rank))
        {
            const double Q_requested = well_pumping_rates[w_id];
            local_requested_pumping_magnitude += std::max(-Q_requested, 0.0);
            const bool well_is_dry =
                (!uo.sim_opt.confined &&
                 global_well_total_screen_sum[w_id] > wet_screen_length_tol &&
                 global_well_wet_screen_sum[w_id] <= wet_screen_length_tol);
            const double Q_at_time = well_is_dry ? 0.0 : Q_requested;

            if (well_is_dry)
            {
                ++local_dry_well_count;
                local_dry_well_requested_total += Q_requested;
                local_pumping_loss_magnitude += std::max(-Q_requested, 0.0);

                if (uo.dry_well_log != 0)
                {
                    const auto &dry_well = mnwells.wells[w_id];
                    const double bottom_head = global_well_bottom_head[w_id];
                    std::ostringstream row;
                    row << std::setprecision(16) << std::scientific
                        << dry_well.Eid << ',' << dry_well.x << ',' << dry_well.y
                        << ',' << dry_well.top << ',' << dry_well.bottom << ',';
                    if (bottom_head > -0.5 * std::numeric_limits<double>::max())
                        row << bottom_head;
                    else
                        row << "nan";
                    row
                        << ',' << time_tracking.simulation_step()
                        << ',' << time_tracking.file_step()
                        << ',' << (current_spinup_active ? 1 : 0)
                        << ',' << current_spinup_solve
                        << ',' << nl_state.nl_iter
                        << ',' << Q_requested;
                    local_dry_well_rows.push_back(row.str());
                }
            }

            block_rhs_vector.block(1)(w_id) += Q_at_time;
            local_well_prescribed_total += Q_at_time;
        }
    }
    finish_assembly_phase("assembling owned-well right-hand sides",
                          assembly_phase_start);

    assembly_phase_start = start_assembly_phase("reducing assembly budget totals");
    {
        const double global_recharge_total = Utilities::MPI::sum(local_recharge_total, mpi_communicator);
        const double global_stream_total = Utilities::MPI::sum(local_stream_total, mpi_communicator);
        const double global_well_prescribed_total = Utilities::MPI::sum(local_well_prescribed_total, mpi_communicator);
        const unsigned int global_dry_well_count = Utilities::MPI::sum(local_dry_well_count, mpi_communicator);
        const double global_dry_well_requested_total = Utilities::MPI::sum(local_dry_well_requested_total, mpi_communicator);
        const double global_requested_pumping_magnitude = Utilities::MPI::sum(local_requested_pumping_magnitude, mpi_communicator);
        const double global_pumping_loss_magnitude = Utilities::MPI::sum(local_pumping_loss_magnitude, mpi_communicator);

        last_recharge_total = global_recharge_total;
        last_stream_total = global_stream_total;
        last_well_total = global_well_prescribed_total;
        last_requested_pumping_magnitude = global_requested_pumping_magnitude;
        last_pumping_loss_magnitude = global_pumping_loss_magnitude;
        last_net_external_total = global_recharge_total + global_stream_total + global_well_prescribed_total;
        last_dry_well_count = global_dry_well_count;

        pcout << "ASM " << std::setw(3) << nl_state.nl_iter
              << " | recharge " << std::scientific << std::setprecision(6) << global_recharge_total
              << " | streams " << global_stream_total
              << " | pumping " << global_well_prescribed_total
              << " | net " << last_net_external_total
              << " | dry wells " << global_dry_well_count
              << std::defaultfloat << std::endl;

        if (uo.verbose_level > 1 && global_dry_well_count > 0)
            pcout << "  Dry-well requested pumping removed: "
                  << std::scientific << global_dry_well_requested_total
                  << std::defaultfloat << std::endl;

    }
    finish_assembly_phase("reducing assembly budget totals",
                          assembly_phase_start);

    if (uo.dry_well_log != 0 && !uo.sim_opt.confined)
    {
        assembly_phase_start = start_assembly_phase("writing dry-well diagnostics");
        const std::string prefix = output_prefix_path();
        const std::string rank_string = Utilities::int_to_string(my_rank, 4);
        const std::string rank_name =
            prefix + "_dry_wells_rank_" + rank_string + ".csv";
        {
            std::ofstream rank_file(rank_name.c_str(), std::ios::out | std::ios::trunc);
            AssertThrow(rank_file.good(),
                        ExcMessage("Could not write dry-well rank snapshot: " + rank_name));
            rank_file
                << "Eid,x,y,top_screen,bot_screen,wt,simulation_counter,input_data_time_step,"
                   "spinup,spinup_solve,nonlinear_iteration,requested_pumping\n";
            for (const auto &row : local_dry_well_rows)
                rank_file << row << '\n';
        }

        MPI_Barrier(mpi_communicator);
        if (my_rank == 0)
        {
            struct DryWellSummary
            {
                double x = 0.0, y = 0.0, top = 0.0, bottom = 0.0;
                double minimum_wt = std::numeric_limits<double>::infinity();
                unsigned long long dry_count = 0;
                unsigned int simulation_counter = 0, input_step = 0;
                unsigned int spinup = 0, spinup_solve = 0, nonlinear_iteration = 0;
            };
            std::map<int, DryWellSummary> summary;
            const std::string summary_name = prefix + "_dry_wells_summary.csv";

            // Preserve the accumulated summary across assemblies and restarts.
            {
                std::ifstream existing(summary_name.c_str());
                std::string line;
                std::getline(existing, line); // header
                while (std::getline(existing, line))
                {
                    if (line.empty())
                        continue;
                    std::replace(line.begin(), line.end(), ',', ' ');
                    std::istringstream values(line);
                    int eid = 0;
                    DryWellSummary entry;
                    if (values >> eid >> entry.x >> entry.y >> entry.top >> entry.bottom
                               >> entry.minimum_wt >> entry.dry_count
                               >> entry.simulation_counter >> entry.input_step
                               >> entry.spinup >> entry.spinup_solve
                               >> entry.nonlinear_iteration)
                        summary[eid] = entry;
                }
            }

            for (unsigned int rank = 0; rank < n_proc; ++rank)
            {
                const std::string source_name =
                    prefix + "_dry_wells_rank_" + Utilities::int_to_string(rank, 4) + ".csv";
                std::ifstream source(source_name.c_str());
                AssertThrow(source.good(),
                            ExcMessage("Could not read dry-well rank snapshot: " + source_name));
                std::string line;
                std::getline(source, line); // header
                while (std::getline(source, line))
                {
                    if (line.empty())
                        continue;
                    std::replace(line.begin(), line.end(), ',', ' ');
                    std::istringstream values(line);
                    int eid = 0;
                    double x = 0.0, y = 0.0, top = 0.0, bottom = 0.0, wt = 0.0;
                    double requested_pumping = 0.0;
                    unsigned int simulation_counter = 0, input_step = 0;
                    unsigned int spinup = 0, spinup_solve = 0, nonlinear_iteration = 0;
                    if (!(values >> eid >> x >> y >> top >> bottom >> wt
                                 >> simulation_counter >> input_step >> spinup
                                 >> spinup_solve >> nonlinear_iteration
                                 >> requested_pumping))
                        continue;

                    DryWellSummary &entry = summary[eid];
                    if (entry.dry_count == 0)
                    {
                        entry.x = x; entry.y = y; entry.top = top; entry.bottom = bottom;
                    }
                    ++entry.dry_count;
                    if (std::isfinite(wt) && wt < entry.minimum_wt)
                    {
                        entry.minimum_wt = wt;
                        entry.simulation_counter = simulation_counter;
                        entry.input_step = input_step;
                        entry.spinup = spinup;
                        entry.spinup_solve = spinup_solve;
                        entry.nonlinear_iteration = nonlinear_iteration;
                    }
                }
            }

            std::ofstream output(summary_name.c_str(), std::ios::out | std::ios::trunc);
            AssertThrow(output.good(),
                        ExcMessage("Could not write dry-well summary: " + summary_name));
            output << "Eid,x,y,top_screen,bot_screen,minimum_wt,dry_count,"
                      "minimum_wt_simulation_counter,minimum_wt_input_data_step,"
                      "minimum_wt_spinup,minimum_wt_spinup_solve,"
                      "minimum_wt_nonlinear_iteration\n"
                   << std::setprecision(16) << std::scientific;
            for (const auto &item : summary)
            {
                const DryWellSummary &entry = item.second;
                output << item.first << ',' << entry.x << ',' << entry.y << ','
                       << entry.top << ',' << entry.bottom << ',';
                if (std::isfinite(entry.minimum_wt))
                    output << entry.minimum_wt;
                else
                    output << "nan";
                output << ',' << entry.dry_count
                       << ',' << entry.simulation_counter << ',' << entry.input_step
                       << ',' << entry.spinup << ',' << entry.spinup_solve
                       << ',' << entry.nonlinear_iteration << '\n';
            }
        }
        MPI_Barrier(mpi_communicator);
        finish_assembly_phase("writing dry-well diagnostics", assembly_phase_start);
    }

    // =====================================================
    // FINALIZE GLOBAL ASSEMBLY
    // =====================================================
    // Compress all matrix blocks
    for (unsigned int i = 0; i < block_system_matrix.n_block_rows(); ++i)
    {
        for (unsigned int j = 0; j < block_system_matrix.n_block_cols(); ++j)
        {
            std::ostringstream phase_name;
            phase_name << "compressing matrix block (" << i << ',' << j << ')';
            assembly_phase_start = start_assembly_phase(phase_name.str());
            block_system_matrix.block(i,j).compress(VectorOperation::add);
            finish_assembly_phase(phase_name.str(), assembly_phase_start);
        }
    }
    for (unsigned int i = 0; i < block_rhs_vector.n_blocks(); ++i)
    {
        std::ostringstream phase_name;
        phase_name << "compressing right-hand-side block " << i;
        assembly_phase_start = start_assembly_phase(phase_name.str());
        block_rhs_vector.block(i).compress(VectorOperation::add);
        finish_assembly_phase(phase_name.str(), assembly_phase_start);
    }
    MPI_Barrier(mpi_communicator);

    if (uo.verbose_level > 1){
        const auto &A00v = block_system_matrix.block(0,0).trilinos_matrix();

        pcout << "lambda_locally_owned_dofs: " << lambda_locally_owned_dofs.n_elements() << "\n";
        pcout << "A00 locally_owned_range_indices: "
              << block_system_matrix.block(0,0).locally_owned_range_indices().n_elements() << "\n";

        pcout << "A00 RangeMap NumMyElements: " << A00v.OperatorRangeMap().NumMyElements() << "\n";

        pcout << "Vector(lambda_locally_owned) ghosts? "
              << TrilinosWrappers::MPI::Vector(lambda_locally_owned_dofs, mpi_communicator).has_ghost_elements()
              << "\n";
        pcout << "Vector(A00 locally_owned_range_indices) ghosts? "
              << TrilinosWrappers::MPI::Vector(block_system_matrix.block(0,0).locally_owned_range_indices(), mpi_communicator).has_ghost_elements()
              << "\n";
    }
}

template <int dim>
void NPSAT_FLOW<dim>::identify_top_active_cells(std::vector<unsigned char> &recharge_receiver,
    std::vector<double> &receiver_recharge_area, std::vector<double> &receiver_effective_z_top) {

    const unsigned int n_local_cells = triangulation.n_locally_owned_active_cells();
    recharge_receiver.assign(n_local_cells, 0);
    receiver_recharge_area.assign(n_local_cells, 0.0);
    receiver_effective_z_top.assign(n_local_cells, std::numeric_limits<double>::quiet_NaN());
    std::unordered_set<std::uint64_t> current_recharge_receiver_gids;

    if (uo.sim_opt.confined) {
        // Confined assembly handles recharge directly on boundary face 5, so
        // no receiver discovery or per-cell receiver data is required here.
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
        AssertThrow(source_area >= 0.0, ExcMessage("Recharge routing received a negative source area."));

        if (trace_cell->is_locally_owned())
        {
            pending_requests.push_back(
                npsat_flow::RechargeRouteRequest{
                    static_cast<std::uint64_t>(trace_cell->global_active_cell_index()),
                    source_area});
            return;
        }

        const unsigned int owner = trace_cell->subdomain_id();
        AssertThrow(owner < n_proc, ExcMessage("Recharge routing reached a remote cell with invalid owner rank."));

        send_requests[owner].push_back(
            npsat_flow::RechargeRouteRequest{
                static_cast<std::uint64_t>(trace_cell->global_active_cell_index()),
                source_area});
    };

    const auto route_local_request = [&](const npsat_flow::RechargeRouteRequest &request)
    {
        const auto cell_it = active_cell_by_gid.find(request.cell_gid);
        AssertThrow(cell_it != active_cell_by_gid.end(), ExcMessage("Recharge routing could not find active cell on owning rank."));

        const TraceCell &trace_cell = cell_it->second;
        AssertThrow(trace_cell->is_locally_owned(), ExcMessage("Recharge routing request was delivered to a rank that does not own the cell."));

        typename DoFHandler<dim>::active_cell_iterator head_cell(&triangulation, trace_cell->level(), trace_cell->index(), &dof_handler_head);
        //const auto head_cell = trace_cell->as_dof_handler_iterator(dof_handler_head);
        head_cell->get_dof_indices(head_dof_indices);

        npsat_flow::CellNonlinearData cell_data;
        compute_cell_r_and_storage(cell_data, head_cell, head_dof_indices);

        //constexpr bool verbose_recharge_routing = false;
        const double drying_saturated_fraction = std::max(0.0, uo.NLC.recharge_drying_saturated_fraction);
        const double wetting_saturated_fraction = std::max(drying_saturated_fraction, uo.NLC.recharge_wetting_saturated_fraction);
        const double min_recharge_relative_k = std::max(0.0, uo.NLC.recharge_min_relative_k);
        const bool was_previous_receiver = previous_recharge_receiver_gids.find(request.cell_gid) != previous_recharge_receiver_gids.end();
        const double selected_saturated_fraction = uo.NLC.use_recharge_hysteresis
            ? (was_previous_receiver ? drying_saturated_fraction : wetting_saturated_fraction) : drying_saturated_fraction;

        const double min_active_saturated_thickness = std::max(r_params.eps, selected_saturated_fraction * cell_data.thickness);
        const bool enough_saturated_thickness = (cell_data.psi > min_active_saturated_thickness);
        const bool enough_relative_k = (cell_data.r > min_recharge_relative_k);
        const bool accepts_recharge = enough_saturated_thickness && enough_relative_k;

        if (uo.verbose_level > 1 && (cell_data.is_partially_saturated || cell_data.is_fully_dry))
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
            receiver_effective_z_top[slot] = std::max(cell_data.h_e, cell_data.z_top - 0.1 * cell_data.thickness); //cell_data.h_e;
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
            AssertThrow(child->is_active(), ExcMessage("Expected active child below refined bottom face."));

            const double child_area = child->face(5)->measure();
            queue_cell(child, child_area);
        }
    };

    constexpr unsigned int top_face = 5;

    for (auto trace_cell = dof_handler_trace.begin_active(); trace_cell != dof_handler_trace.end(); ++trace_cell){
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
        AssertThrow(ierr == MPI_SUCCESS, ExcMessage("MPI_Alltoallv failed during recharge routing."));

        const unsigned int local_pending = static_cast<unsigned int>(pending_requests.size());
        const unsigned int global_pending = Utilities::MPI::sum(local_pending, mpi_communicator);

        if (global_pending == 0)
            break;
    }

    previous_recharge_receiver_gids.swap(current_recharge_receiver_gids);
}

template <int dim>
void NPSAT_FLOW<dim>::compute_cell_r_and_storage(npsat_flow::CellNonlinearData &out,
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    const std::vector<types::global_dof_index> &head_dof_indices, const double effective_z_top) const {

    // ---------------------------
    // (1) Compute z_bot / z_top from cell vertices
    // ---------------------------
    double z_top = 0.0;
    double z_bot = 0.0;
    const auto bot_face = cell->face(4);
    const auto top_face = cell->face(5);
    for (unsigned int fv = 0; fv < GeometryInfo<dim>::vertices_per_face; ++fv)
    {
        z_bot += bot_face->vertex(fv)[dim-1];
        z_top += top_face->vertex(fv)[dim-1];
    }
    z_bot /= static_cast<double>(GeometryInfo<dim>::vertices_per_face);
    z_top /= static_cast<double>(GeometryInfo<dim>::vertices_per_face);
    out.z_bot = z_bot;
    out.z_top = z_top;
    //std::cout << "Rank: " << my_rank << " computes CELL Z: " << z_bot << " " << z_top << std::endl;
    Assert(out.z_top >= out.z_bot, ExcMessage("Top face is below bottom face. Check face ids."));

    // ---------------------------
    // (2) Representative head h_e (DG0: one DoF per cell)
    // ---------------------------
    Assert(head_dof_indices.size() >= 1, ExcInternalError());
    const types::global_dof_index hdof = head_dof_indices[0];
    // Important: h_guess must have updated ghosts before calling this function.
    out.h_e = h_guess[hdof];
    // State variable: water column above bottom (NOT thickness)
    out.psi = out.h_e - out.z_bot;

    const bool has_effective_top = std::isfinite(effective_z_top) && effective_z_top > out.z_bot;
    out.uses_effective_top = has_effective_top;
    out.assembly_z_top = has_effective_top ? effective_z_top : out.z_top;
    Assert(out.assembly_z_top >= out.z_bot, ExcMessage("Assembly top elevation is below cell bottom."));

    const double geometric_thickness = out.z_top - out.z_bot;
    out.thickness = geometric_thickness;

    // ---------------------------
    // (3) Saturation classification (with a small tolerance)
    //
    // Classify the physical cell against the geometric top.  The optional
    // effective top belongs to recharge routing/stabilization; it must not make
    // a physically partially saturated cell look saturated for transmissivity.
    // ---------------------------
    constexpr double tol = 1e-12;
    out.is_fully_dry       = (out.h_e <= out.z_bot + tol);
    out.is_fully_saturated = (out.h_e >= out.z_top - tol);
    out.is_partially_saturated = (!out.is_fully_dry && !out.is_fully_saturated);

    // ---------------------------
    // (4) Identify whether this cell is in the top layer (boundary top face)
    // ---------------------------
    out.is_top_layer_cell = cell->face(5)->at_boundary();

    if (uo.sim_opt.confined)
    {
        out.uses_effective_top = false;
        out.assembly_z_top = out.z_top;
        out.is_fully_dry = false;
        out.is_partially_saturated = false;
        out.is_fully_saturated = true;
        out.r = 1.0;

        const Point<dim> p = cell->center();
        out.S_eff = hgeo_prop.specific_storage(p);
        return;
    }

    // ---------------------------
    // (5) Compute r(h) from physical transmissive thickness.
    //
    // local_A integrates over the full geometric cell.  Therefore r must scale
    // K so that K_eff * geometric_thickness represents the intended
    // transmissivity:
    //
    //   dry:               r = r_min
    //   partial:           r = (h - z_bot) / (z_top - z_bot)
    //   saturated interior:r = 1
    //   saturated top cell:r = (h - z_bot) / (z_top - z_bot), allowed > 1
    //
    // The optional effective top is intentionally not used here.  It is a
    // recharge-routing stabilization height, not the physical element height.
    // ---------------------------

    const double b = std::max(geometric_thickness, 1e-12);
    const double r_min = r_params.r_min;

    const double transmissive_fraction_raw = (out.h_e - out.z_bot) / b;
    const double theta = std::max(0.0, std::min(1.0, transmissive_fraction_raw));

    double r = 1.0;
    if (out.is_fully_dry)
    {
        r = r_min;
    }
    else if (out.is_partially_saturated)
    {
        r = std::max(r_min, transmissive_fraction_raw);
    }
    else if (out.is_top_layer_cell)
    {
        r = std::max(r_min, transmissive_fraction_raw);
    }
    else
    {
        r = 1.0;
    }

    r = std::max(r_min, r);
    out.r = r;

    // ---------------------------
    // (6) Compute effective storage coefficient S_eff for this cell (scalar).
    //
    // The model does not solve the unsaturated zone. Storage is therefore tied
    // to saturated thickness:
    //   h <= z_bot: dry cell, no drainable saturated storage
    //   z_bot < h < z_top: water table in cell, add Sy over this cell thickness
    //   h >= z_top: fully saturated cell, confined compressive storage only
    //
    // Since local_M integrates over the full geometric cell volume, Ss is
    // scaled by the geometric saturated-volume fraction theta. Sy is converted
    // to a volumetric coefficient Sy/b and smoothly windowed to the geometric
    // water-table cell.
    // ---------------------------

    const Point<dim> p = cell->center();
    const double Ss = hgeo_prop.specific_storage(p);// [1/L]
    const double Sy = hgeo_prop.specific_yield(p);// [-]

    // 3D-consistent volumetric coefficients over the full cell volume.
    const double S_conf_vol   = Ss * theta; // [1/L]
    const double S_unconf_vol = Sy / b;     // [1/L]

    // Storage smoothing thickness scale.
    const double epsS = std::max(r_params.eps, 1e-12);

    // Window active only when the head is between the geometric bottom and top.
    const double w_bot = npsat_flow::logistic_sigma((out.h_e - out.z_bot) / epsS);
    // For effective-top cells retain the unconfined storage window up to the
    // extended top. This is equivalent to extending the top cell to the head,
    // while r>1 supplies the corresponding additional transmissive thickness.
    const double w_top = npsat_flow::logistic_sigma((out.assembly_z_top - out.h_e) / epsS);
    const double w = w_bot * w_top;


    out.S_eff = S_conf_vol + w * S_unconf_vol;
    //out.S_eff = Ss;
}

template <int dim>
double NPSAT_FLOW<dim>::effective_top_for_cell(
    const npsat_flow::CellNonlinearData &cell_data,
    const double routed_receiver_effective_z_top) const {
    using EffectiveTopMode = npsat_flow::NonlinearControls::EffectiveTopMode;
    using RechargeStabilizationMode = npsat_flow::NonlinearControls::RechargeStabilizationMode;

    if (uo.sim_opt.confined)
        return std::numeric_limits<double>::quiet_NaN();

    if (uo.NLC.recharge_stabilization_mode == RechargeStabilizationMode::HysteresisOnly || uo.NLC.effective_top_mode == EffectiveTopMode::Off)
        return std::numeric_limits<double>::quiet_NaN();

    const auto valid_effective_top = [&](const double z) -> bool
    {
        return std::isfinite(z) && z > cell_data.z_bot + 1e-12;
    };

    if (valid_effective_top(routed_receiver_effective_z_top))
        return routed_receiver_effective_z_top;

    if (uo.NLC.effective_top_mode != EffectiveTopMode::AllWaterTableCells)
        return std::numeric_limits<double>::quiet_NaN();

    constexpr double tol = 1e-12;
    const bool has_water_above_bottom = cell_data.h_e > cell_data.z_bot + tol;
    const bool is_geometric_water_table_cell = has_water_above_bottom && (cell_data.h_e < cell_data.z_top - tol || cell_data.is_top_layer_cell);

    if (is_geometric_water_table_cell && valid_effective_top(cell_data.h_e))
        return cell_data.h_e;

    return std::numeric_limits<double>::quiet_NaN();
}

template <int dim>
bool NPSAT_FLOW<dim>::should_receive_gw_recharge(const std::vector<unsigned char> &recharge_receiver,
    const typename DoFHandler<dim>::active_cell_iterator &cell, const unsigned int i_face) const {
    AssertThrow(i_face < GeometryInfo<dim>::faces_per_cell,
                ExcMessage("Invalid face index for recharge decision."));

    if (i_face != 5)
        return false;

    if (uo.sim_opt.confined)
        return cell->face(i_face)->at_boundary();

    const unsigned int slot = cell->user_index();
    AssertIndexRange(slot, recharge_receiver.size());

    return recharge_receiver[slot] != 0;
}

template <int dim>
void NPSAT_FLOW<dim>::print_matrix(const FullMatrix<double> &A, const std::string& name)
{
    if (uo.print_matrices)
    {
        pcout<< "------------ " << name << "---------------\n";
        A.print(pcout.get_stream(), 8, 2);
        pcout<< "------------------------------------------\n";
    }
}

template <int dim>
void NPSAT_FLOW<dim>::print_vector(const Vector<double> &A, const std::string& name)
{
    if (uo.print_matrices)
    {
        pcout<< "------------ " << name << "---------------\n";
        A.print(pcout.get_stream());
        pcout<< "------------------------------------------\n";
    }
}



#endif //NPSAT_FLOW_ASSEMBLE_IMPL_H
