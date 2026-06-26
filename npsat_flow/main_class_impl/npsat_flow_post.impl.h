//
// Created by giorgk on 6/25/26.
//

#ifndef NPSAT_FLOW_POST_IMPL_H
#define NPSAT_FLOW_POST_IMPL_H

template <int dim>
void NPSAT_FLOW<dim>::compute_heads(){
    pcout << "Recovering heads from trace solution..." << std::endl;
    h_new = 0.0;
    double delta_time = time_tracking.duration();

    // Preallocate small local objects once (RT0/DG0 fixed sizes)
    const unsigned int n_trace_dofs      = fe_trace.n_dofs_per_cell();
    const unsigned int n_head_dofs       = fe_head.n_dofs_per_cell();
    Vector<double> lambda_local(6);

    FullMatrix<double> E_matrix(1, 6);
    std::vector<types::global_dof_index> trace_dof_indices(n_trace_dofs);
    std::vector<types::global_dof_index> head_dof_indices(n_head_dofs);

    double local_head_min = std::numeric_limits<double>::max();
    double local_head_max = -std::numeric_limits<double>::max();
    double local_head_sum = 0.0;
    unsigned int local_head_count = 0;

    double local_abs_head_max = -1.0;
    double outlier_head = 0.0;
    double outlier_kinv = 0.0;
    double outlier_V = 0.0;
    double outlier_E_lambda = 0.0;
    double outlier_rhs = 0.0;
    double outlier_lambda_min = 0.0;
    double outlier_lambda_max = 0.0;
    unsigned int outlier_active_cell_index = numbers::invalid_unsigned_int;
    unsigned int outlier_slot = numbers::invalid_unsigned_int;
    types::global_dof_index outlier_hdof = numbers::invalid_dof_index;
    Point<dim> outlier_center;
    npsat_flow::CellNonlinearData outlier_nl_data;
    bool outlier_has_well = false;

    for (auto head_cell = dof_handler_head.begin_active(); head_cell != dof_handler_head.end(); ++head_cell){
        if (!head_cell->is_locally_owned())
            continue;

        head_cell->get_dof_indices(head_dof_indices);
        const auto hdof = head_dof_indices[0];
        Assert(head_locally_owned_dofs.is_element(hdof), ExcMessage("DG0 head DoF on locally owned cell is not locally owned."));

        const unsigned int slot = head_cell->user_index();
        Assert(slot != numbers::invalid_unsigned_int, ExcInternalError());
        local_element_data_rt_0dg0.assert_valid_slot(slot);

        // Get corresponding trace cell iterator (same triangulation, different DoFHandler)
        //const auto trace_cell = head_cell->as_dof_handler_iterator(dof_handler_trace);
        typename DoFHandler<dim>::active_cell_iterator trace_cell(&triangulation, head_cell->level(), head_cell->index(), &dof_handler_trace);

        // Fetch local trace dof indices (6 for RT0 trace)
        trace_cell->get_dof_indices(trace_dof_indices);
        npsat_flow::CellNonlinearData nl_cell_data;
        compute_cell_r_and_storage(nl_cell_data, head_cell, head_dof_indices);

        for (unsigned int i = 0; i < n_trace_dofs; ++i)
        {
            lambda_local(i) = solution_trace[trace_dof_indices[i]];
        }
        double lambda_min = lambda_local(0);
        double lambda_max = lambda_local(0);
        for (unsigned int i = 1; i < n_trace_dofs; ++i)
        {
            lambda_min = std::min(lambda_min, lambda_local(i));
            lambda_max = std::max(lambda_max, lambda_local(i));
        }

        // Load cached E (1x6), Kinv (scalar), and V (scalar)
        local_element_data_rt_0dg0.get_E(slot, E_matrix);
        const double kinv = local_element_data_rt_0dg0.get_kinv(slot);
        const double V    = local_element_data_rt_0dg0.get_V(slot);

        // -----------------------------------------------------------------
        // No-well RHS: rhs = V - dt * (E * lambda)
        // where V = M*h_old + dt*F (already stored)
        // -----------------------------------------------------------------
        double E_lambda = 0.0;
        for (unsigned int j = 0; j < 6; ++j)
            E_lambda += E_matrix(0, j) * lambda_local(j);

        //rhs = data.V_vector;    // V = M*h_old + Δt*F
        //rhs -= E_lambda;        // V - E*Λ
        double rhs = V - delta_time * E_lambda;

        // -----------------------------------------------------------------
        // MNW correction (DG0-friendly):
        // If this cell has well links, the head recovery becomes:
        //
        //   (K + dt*sum_cwc) * H = rhs + dt * sum_cwc_hw
        //
        // where sum_cwc    = Σ_e CWC_e
        //       sum_cwc_hw = Σ_e CWC_e * h_w
        //
        // For DG0: K is scalar K(0,0), rhs is scalar rhs(0).
        // -----------------------------------------------------------------
        // Find wells in this cell (local_cell_well_map is sorted by cell id)
        const unsigned int cell_id = head_cell->active_cell_index();
        auto it = std::lower_bound(
            local_cell_well_map.begin(), local_cell_well_map.end(),
            cell_id,
            [](const std::pair<unsigned int, std::vector<npsat_flow::CellWellLink>> &pair,
               unsigned int val)
            { return pair.first < val; });

        double h_local = 0.0;
        const bool has_well = (it != local_cell_well_map.end() && it->first == cell_id);
        if (it != local_cell_well_map.end() && it->first == cell_id){
            double sum_cwc    = 0.0;
            double sum_cwc_hw = 0.0;

            // Loop the well links for this cell
            for (const auto &link : it->second)
            {
                const double cwc =npsat_flow::effective_well_link_conductance(link, nl_cell_data.h_e, uo.sim_opt.confined);
                const unsigned int w_id = mnwells.wells[link.well_global_index].global_index;

                sum_cwc    += cwc;
                sum_cwc_hw += cwc * well_solution_ghosted[w_id];
            }
            // K is scalar for DG0
            const double K_scalar = kinv != 0.0 ? (1.0 / kinv) : 0.0;
            // Modify rhs and denominator
            rhs += delta_time * sum_cwc_hw;

            const double denom = K_scalar + delta_time * sum_cwc;

            // Recover head (scalar)
            h_local = rhs / denom;

            // Copy to global vector
            h_new[hdof] = h_local;
        }
        else
        {
            // -----------------------------------------------------------------
            // No-well recovery: h_local = K^{-1} * rhs
            // -----------------------------------------------------------------
            h_local = kinv * rhs;
            h_new[hdof] = h_local;
        }

        local_head_min = std::min(local_head_min, h_local);
        local_head_max = std::max(local_head_max, h_local);
        local_head_sum += h_local;
        ++local_head_count;

        const double abs_h = std::abs(h_local);
        if (abs_h > local_abs_head_max)
        {
            local_abs_head_max = abs_h;
            outlier_head = h_local;
            outlier_kinv = kinv;
            outlier_V = V;
            outlier_E_lambda = E_lambda;
            outlier_rhs = rhs;
            outlier_lambda_min = lambda_min;
            outlier_lambda_max = lambda_max;
            outlier_active_cell_index = head_cell->active_cell_index();
            outlier_slot = slot;
            outlier_hdof = hdof;
            outlier_center = head_cell->center();
            outlier_has_well = has_well;
            compute_cell_r_and_storage(outlier_nl_data, head_cell, head_dof_indices);
        }
    }

    MPI_Barrier(mpi_communicator);
    // Update ghost values for parallel
    h_new.compress(VectorOperation::insert);

    const double global_min = Utilities::MPI::min(local_head_min, mpi_communicator);
    const double global_max = Utilities::MPI::max(local_head_max, mpi_communicator);
    const double global_sum = Utilities::MPI::sum(local_head_sum, mpi_communicator);
    const unsigned int global_count = Utilities::MPI::sum(local_head_count, mpi_communicator);
    const double global_mean = (global_count > 0 ? global_sum / static_cast<double>(global_count) : 0.0);
    const double global_abs_head_max =
        Utilities::MPI::max(local_abs_head_max, mpi_communicator);

    pcout << "  Head range: [" << global_min << ", " << global_max
          << "] m, Mean: " << global_mean << " m" << std::endl;

    if (uo.verbose_level > 1 &&std::abs(local_abs_head_max - global_abs_head_max) <= 1e-12 * std::max(1.0, global_abs_head_max))
    {
        std::cout << std::setprecision(16)
                  << "  Head recovery max |h| diagnostic on rank " << my_rank << ":\n"
                  << "    cell_active_index=" << outlier_active_cell_index
                  << " slot=" << outlier_slot
                  << " hdof=" << outlier_hdof
                  << " center=" << outlier_center
                  << " has_well=" << outlier_has_well << "\n"
                  << "    h_recovered=" << outlier_head
                  << " h_guess=" << outlier_nl_data.h_e
                  << " z_bot=" << outlier_nl_data.z_bot
                  << " z_top=" << outlier_nl_data.z_top
                  << " r=" << outlier_nl_data.r
                  << " S_eff=" << outlier_nl_data.S_eff << "\n"
                  << "    dry=" << outlier_nl_data.is_fully_dry
                  << " partial=" << outlier_nl_data.is_partially_saturated
                  << " saturated=" << outlier_nl_data.is_fully_saturated
                  << " top_layer=" << outlier_nl_data.is_top_layer_cell << "\n"
                  << "    kinv=" << outlier_kinv
                  << " V=" << outlier_V
                  << " E_lambda=" << outlier_E_lambda
                  << " rhs=V-dt*E_lambda=" << outlier_rhs
                  << " lambda_range=[" << outlier_lambda_min
                  << ", " << outlier_lambda_max << "]"
                  << std::endl;
    }

    TrilinosWrappers::MPI::Vector diff(h_new);
    diff -= h_guess;
    pcout << "  Head update L2 norm = " << diff.l2_norm() << std::endl;
}

template <int dim>
void NPSAT_FLOW<dim>::compute_update_norm(const TrilinosWrappers::MPI::Vector &h_prev,
                               const TrilinosWrappers::MPI::Vector &h_next,
                               double &update_norm, double &ref_norm) const{
    AssertDimension(h_prev.size(), h_next.size());

    // We compute an L-infinity (max) norm of the update on locally-owned entries,
    // then take the global max across MPI ranks.
    //
    // Rationale:
    // - robust and cheap
    // - independent of mesh size scaling
    // - suitable for nonlinear Picard/Anderson stopping criteria

    double local_update_max = 0.0;
    double local_ref_max    = 0.0;
    double local_signed_update_at_max = 0.0;
    double local_h_prev_at_max = 0.0;
    double local_h_next_at_max = 0.0;
    unsigned int local_update_cell_index = numbers::invalid_unsigned_int;
    unsigned int local_update_slot = numbers::invalid_unsigned_int;
    types::global_dof_index local_update_hdof = numbers::invalid_dof_index;
    Point<dim> local_update_center;
    npsat_flow::CellNonlinearData local_update_cell_data;

    const unsigned int n_head_dofs = fe_head.n_dofs_per_cell();
    std::vector<types::global_dof_index> head_dof_indices(n_head_dofs);

    for (auto head_cell = dof_handler_head.begin_active(); head_cell != dof_handler_head.end(); ++head_cell){
        if (!head_cell->is_locally_owned())
            continue;

        head_cell->get_dof_indices(head_dof_indices);
        const auto i = head_dof_indices[0];
        const double dh = std::abs(h_next[i] - h_prev[i]);
        const double h  = std::abs(h_next[i]);
        if (dh > local_update_max)
        {
            local_update_max = dh;
            local_signed_update_at_max = h_next[i] - h_prev[i];
            local_h_prev_at_max = h_prev[i];
            local_h_next_at_max = h_next[i];
            local_update_cell_index = head_cell->active_cell_index();
            local_update_slot = head_cell->user_index();
            local_update_hdof = i;
            local_update_center = head_cell->center();
            compute_cell_r_and_storage(local_update_cell_data, head_cell, head_dof_indices);
        }
        local_ref_max    = std::max(local_ref_max, h);
        // Optional more robust scaling:
        // local_ref_max = std::max(local_ref_max, std::abs(hp));
    }
    // Global reductions
    update_norm = Utilities::MPI::max(local_update_max, mpi_communicator);
    ref_norm    = Utilities::MPI::max(local_ref_max,    mpi_communicator);

    // Safety: avoid zero reference scale downstream
    if (ref_norm < 1e-30)
        ref_norm = 1.0;

    if (uo.verbose_level > 1 && (std::abs(local_update_max - update_norm) <=
        1e-12 * std::max(1.0, update_norm)))
    {
        std::cout << std::setprecision(16)
                  << "  Head update max diagnostic on rank " << my_rank << ":\n"
                  << "    cell_active_index=" << local_update_cell_index
                  << " slot=" << local_update_slot
                  << " hdof=" << local_update_hdof
                  << " center=" << local_update_center << "\n"
                  << "    h_prev=" << local_h_prev_at_max
                  << " h_next=" << local_h_next_at_max
                  << " signed_update=" << local_signed_update_at_max
                  << " abs_update=" << local_update_max << "\n"
                  << "    z_bot=" << local_update_cell_data.z_bot
                  << " z_top=" << local_update_cell_data.z_top
                  << " psi=" << local_update_cell_data.psi
                  << " r=" << local_update_cell_data.r
                  << " S_eff=" << local_update_cell_data.S_eff << "\n"
                  << "    dry=" << local_update_cell_data.is_fully_dry
                  << " partial=" << local_update_cell_data.is_partially_saturated
                  << " saturated=" << local_update_cell_data.is_fully_saturated
                  << " top_layer=" << local_update_cell_data.is_top_layer_cell
                  << std::endl;
    }

}

template <int dim>
bool NPSAT_FLOW<dim>::check_nonlinear_convergence(const double update_norm, const double ref_norm) const{
    // Protect against a zero/negative reference norm
    const double ref = std::max(ref_norm, 1e-30);

    // Mixed absolute + relative stopping threshold
    const double threshold = uo.NLC.abs_tol_update  + uo.NLC.rel_tol_update * ref;

    // update_norm is assumed to already be a *global* max norm
    // (as returned by compute_update_norm())
    const bool converged = (update_norm <= threshold);

    pcout << "+--------------------------------------+\n"
          << "| update_norm : " << std::fixed << std::setprecision(6)
          << std::setw(12) << update_norm << " |\n"
          << "| threshold   : "
          << std::setw(12) << threshold << " |\n"
          << "+--------------------------------------+\n"
          << "| Status      : "
          << (converged ? "CONVERGED     " : "NOT CONVERGED ")
          << "|\n"
          << "+--------------------------------------+"
          << std::defaultfloat << std::endl;

    return converged;
}

template <int dim>
bool NPSAT_FLOW<dim>::anderson_accelerate(TrilinosWrappers::MPI::Vector &x_accel,
                                    const TrilinosWrappers::MPI::Vector &x_k,
                                    const TrilinosWrappers::MPI::Vector &G_xk,
                                    npsat_flow::NonlinearState &nl_state,
                                    const npsat_flow::NonlinearControls &ctl) const{

    AssertThrow(ctl.anderson_m >= 1, ExcMessage("Anderson memory depth must be >= 1."));

    // Ensure output has correct layout
    x_accel = G_xk;

    // f_k = G(x_k) - x_k
    TrilinosWrappers::MPI::Vector f_k = G_xk;
    f_k -= x_k;

    // -------------------------
    // (1) Always update history
    // -------------------------
    nl_state.x_hist.push_back(x_k);
    nl_state.f_hist.push_back(f_k);

    // Keep at most (m+1) items so we can form m deltas
    const std::size_t max_keep = static_cast<std::size_t>(ctl.anderson_m) + 1;
    while (nl_state.x_hist.size() > max_keep)
        nl_state.x_hist.pop_front();
    while (nl_state.f_hist.size() > max_keep)
        nl_state.f_hist.pop_front();

    // ---------------------------------------------
    // (2) Skip AA usage until we reach anderson_start
    //     BUT keep the stored history above.
    // ---------------------------------------------
    if (nl_state.nl_iter < ctl.anderson_start)
        return false;

    // Need at least two history entries to form one delta.
    const std::size_t L = nl_state.x_hist.size();
    if (L < 2)
        return false; // not enough to form a single delta

    const unsigned int m_used = std::min<unsigned int>(ctl.anderson_m, static_cast<unsigned int>(L - 1));

    if (m_used == 0)
        return false;

    // Build delta vectors for the last m_used steps:
    // Use the most recent (m_used) deltas ending at the latest entry.
    std::vector<TrilinosWrappers::MPI::Vector> dx(m_used);
    std::vector<TrilinosWrappers::MPI::Vector> df(m_used);

    // Indices in deque: [0 ... L-1], latest is L-1
    // We want deltas for pairs: (L-m_used-1 -> L-m_used), ..., (L-2 -> L-1)
    const std::size_t start = L - 1 - m_used;

    for (unsigned int i = 0; i < m_used; ++i)
    {
        const std::size_t j0 = start + i;
        const std::size_t j1 = j0 + 1;

        dx[i] = nl_state.x_hist[j1];
        dx[i] -= nl_state.x_hist[j0];

        df[i] = nl_state.f_hist[j1];
        df[i] -= nl_state.f_hist[j0];
    }

    // Assemble normal equations: (Df^T Df + reg I) alpha = Df^T f_k
    LAPACKFullMatrix<double> A(m_used, m_used);
    Vector<double> b(m_used);

    for (unsigned int i = 0; i < m_used; ++i)
    {
        // RHS: <df_i, f_k>
        b[i] = df[i] * f_k;

        for (unsigned int j = 0; j < m_used; ++j)
            A(i, j) = df[i] * df[j];
    }

    // Regularization (helps when columns nearly dependent)
    const double reg = std::max(ctl.anderson_reg, 0.0);
    if (reg > 0.0)
        for (unsigned int i = 0; i < m_used; ++i)
            A(i, i) += reg;

    // Solve for alpha
    try
    {
        A.compute_lu_factorization();
        A.solve(b); // b becomes alpha
    }
    catch (...)
    {
        return false;
    }

    // Sanity check on alpha values
    for (unsigned int i = 0; i < m_used; ++i)
        if (!std::isfinite(b[i]))
            return false;

    // Form accelerated iterate: x_accel = G_xk - sum_i alpha_i * dx_i
    // (Recall x_accel currently equals G_xk)
    for (unsigned int i = 0; i < m_used; ++i)
        x_accel.add(-b[i], dx[i]);

    return true;
}

template <int dim>
void NPSAT_FLOW<dim>::apply_damped_update(TrilinosWrappers::MPI::Vector &h_guess,
                                     const TrilinosWrappers::MPI::Vector &h_candidate,
                                     const double omega) const{
    AssertDimension(h_guess.size(), h_candidate.size());
    AssertThrow(omega >= 0.0 && omega <= 1.0,
                ExcMessage("apply_damped_update(): omega must be in [0,1]."));

    // Update only locally owned entries. Ghost entries (if present) are ignored here and
    // should be updated by the caller via h_guess.update_ghost_values() if needed.
    for (auto it = head_locally_owned_dofs.begin(); it != head_locally_owned_dofs.end(); ++it)
    {
        const auto i = *it;
        h_guess[i] = (1.0 - omega) * h_guess[i] + omega * h_candidate[i];
    }
}

template <int dim>
void NPSAT_FLOW<dim>::compute_fluxes(){
    pcout << "Recovering fluxes from trace and head solutions..." << std::endl;

    TrilinosWrappers::MPI::Vector q_owned;
    q_owned.reinit(flux_locally_owned_dofs, mpi_communicator);
    q_owned = 0.0;

    q_new = 0;

    const unsigned int n_trace_dofs      = fe_trace.n_dofs_per_cell();
    const unsigned int n_head_dofs       = fe_head.n_dofs_per_cell();
    const unsigned int n_flux_dofs       = fe_flux.n_dofs_per_cell();

    std::vector<types::global_dof_index> trace_dof_indices(n_trace_dofs);
    std::vector<types::global_dof_index> head_dof_indices(n_head_dofs);
    std::vector<types::global_dof_index> flux_dof_indices(n_flux_dofs);

    Vector<double> lambda_local(n_trace_dofs);
    double         h_local = 0.0;

    // Cached matrices
    FullMatrix<double> A_inv(n_flux_dofs, n_flux_dofs);
    FullMatrix<double> C_mat(n_trace_dofs, n_flux_dofs);
    FullMatrix<double> B_mat(n_head_dofs, n_flux_dofs);

    // For well exchange
    FullMatrix<double> ae_matrix(n_head_dofs, n_trace_dofs);

    // Local flux vectors
    Vector<double> rhs_flux(n_flux_dofs);
    Vector<double> q_local(n_flux_dofs);

    // For each cell, recover local flux
    // Iterate on any DoFHandler; use triangulation cell to map to others
    for (const auto &flux_cell : dof_handler_flux.active_cell_iterators()){
        if (!flux_cell->is_locally_owned())
            continue;

        const auto tria_cell = flux_cell;

        const unsigned int slot = tria_cell->user_index();
        Assert(slot != numbers::invalid_unsigned_int, ExcInternalError());
        local_element_data_rt_0dg0.assert_valid_slot(slot);

        // Map to the three DoFHandlers (same underlying triangulation)
        typename DoFHandler<dim>::active_cell_iterator trace_cell(&triangulation, tria_cell->level(), tria_cell->index(), &dof_handler_trace);
        typename DoFHandler<dim>::active_cell_iterator head_cell(&triangulation, tria_cell->level(), tria_cell->index(), &dof_handler_head);
        //const auto trace_cell = tria_cell->as_dof_handler_iterator(dof_handler_trace);
        //const auto head_cell  = tria_cell->as_dof_handler_iterator(dof_handler_head);

        // Get dof indices
        trace_cell->get_dof_indices(trace_dof_indices);
        flux_cell->get_dof_indices(flux_dof_indices);
        head_cell->get_dof_indices(head_dof_indices);
        npsat_flow::CellNonlinearData nl_cell_data;
        compute_cell_r_and_storage(nl_cell_data, head_cell, head_dof_indices);

        // Get Λ values for this cell
        for (unsigned int i = 0; i < n_trace_dofs; ++i)
        {
            lambda_local(i) = solution_trace[trace_dof_indices[i]];
        }

        // Get head values for this cell
        // Gather head (DG0 scalar)
        h_local = h_new[head_dof_indices[0]];

        // Load cached postprocess matrices A_inv, B, C
        local_element_data_rt_0dg0.get_A_inv(slot, A_inv);  // 6x6
        local_element_data_rt_0dg0.get_B(slot,     B_mat);  // 1x6
        local_element_data_rt_0dg0.get_C(slot,     C_mat);  // 6x6

        // ----------------------------------------------------
        // rhs_flux = -(B^T * h + C^T * lambda)
        // DG0: B is 1x6, so (B^T*h)_i = B(0,i) * h
        // ----------------------------------------------------
        for (unsigned int i = 0; i < n_flux_dofs; ++i)
        {
            double BT_h_i = B_mat(0, i) * h_local;

            // (C^T * lambda)_i = sum_j C(j,i) * lambda(j)
            double CT_lambda_i = 0.0;
            for (unsigned int j = 0; j < n_trace_dofs; ++j)
                CT_lambda_i += C_mat(j, i) * lambda_local(j);

            rhs_flux(i) = -(BT_h_i + CT_lambda_i);
        }

        // ---------------------------------------
        // q = A⁻¹ rhs
        // ---------------------------------------
        //Vector<double> q_local(n_flux_dofs);
        A_inv.vmult(q_local, rhs_flux);

        // Step 5: Copy to global vector // Scatter to global q_new
        for (unsigned int i = 0; i < n_flux_dofs; ++i)
        {
            q_owned[flux_dof_indices[i]] = q_local(i);
        }

        // Calculate the Cell - Well exchange
        // ----------------------------------------------------
        // Cell-well exchange postprocess (optional)
        // Uses:
        //  ae_matrix (1x6) and H_known = Kinv_V (scalar)
        // ----------------------------------------------------
        {
            auto it = std::lower_bound(local_cell_well_map.begin(), local_cell_well_map.end(),
                  trace_cell->active_cell_index(),
                  [](const std::pair<unsigned int, std::vector<npsat_flow::CellWellLink>>& pair, unsigned int val) {
                                 return pair.first < val;});

            if (it != local_cell_well_map.end() && it->first == trace_cell->active_cell_index()){
                // If the cell contains wells
                // Load cached ae (1x6) and H_known (=Kinv_V)
                local_element_data_rt_0dg0.get_ae(slot, ae_matrix);         // 1x6
                const double H_known = local_element_data_rt_0dg0.get_kinv_v(slot); // scalar

                double alpha_dot_lambda = 0.0;
                for (unsigned int i = 0; i < n_trace_dofs; ++i)
                    alpha_dot_lambda += ae_matrix(0, i) * lambda_local(i);

                //const double H_known = data.Kinv_V(0);
                const double h_e = H_known - alpha_dot_lambda;

                // Update links
                for (auto& link : it->second) {
                    const unsigned int w_id = mnwells.wells[link.well_global_index].global_index;
                    const double hw = well_solution_ghosted[w_id];
                    const double cwc =
                        npsat_flow::effective_well_link_conductance(link,
                                                                  nl_cell_data.h_e,
                                                                  uo.sim_opt.confined);
                    //const double Qe_cond = cwc * (hw - h_e);
                    //const double Qe_test = cwc * (hw - h_local);
                    link.cwc_eff = cwc;
                    link.Qe = cwc * (hw - h_e);  // positive = injection, negative = pumping
                    link.h_e = h_e;
                    // pcout << "Well: " << link.well_global_index << ", "
                    //         << cwc << ", hw=" << hw
                    //         << ", He_cond=" << h_e
                    //         << ", h_local=" << h_local
                    //         << ", sl=" << link.sl
                    //         << ", ze=" << link.ze
                    //         << ", Qe_cond=" << Qe_cond
                    //         << ", Qe_test=" << Qe_test << std::endl;
                }
            }
        }
    }
    // Update ghost values for parallel
    q_owned.compress(VectorOperation::insert);

    q_new = q_owned;
    q_new.update_ghost_values();

}

#endif //NPSAT_FLOW_POST_IMPL_H
