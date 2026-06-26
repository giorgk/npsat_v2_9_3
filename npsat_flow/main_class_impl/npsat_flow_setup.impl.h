//
// Created by giorgk on 6/24/26.
//

#ifndef NPSAT_FLOW_SETUP_IMPL_H
#define NPSAT_FLOW_SETUP_IMPL_H

template <int dim>
void NPSAT_FLOW<dim>::setup_system() {

    pcout << "=======================================================" << std::endl;
    pcout << "Setup system..." << std::endl << std::endl;

    pcout << "Distributing DoFs..." << std::endl;

    {
        const unsigned int n_local = triangulation.n_locally_owned_active_cells();
        std::vector<unsigned char> seen(n_local, 0);
        for (auto cell = triangulation.begin_active(); cell != triangulation.end(); ++cell)
        {
            if (!cell->is_locally_owned())
                continue;

            const unsigned int slot = cell->user_index();
            AssertThrow(slot < n_local, ExcMessage("Cell user_index slots are not initialized after refinement."));
            AssertThrow(seen[slot] == 0, ExcMessage("Duplicate cell user_index slot detected."));
            seen[slot] = 1;
        }
    }

    // 1. Distribute DoFs for each space
    dof_handler_flux.distribute_dofs(fe_flux);
    dof_handler_head.distribute_dofs(fe_head);
    dof_handler_trace.distribute_dofs(fe_trace);

    pcout << "Number of flux DoFs: " << dof_handler_flux.n_dofs() << std::endl;
    pcout << "Number of head DoFs: " << dof_handler_head.n_dofs() << std::endl;
    pcout << "Number of trace DoFs: " << dof_handler_trace.n_dofs() << std::endl;

    // 2. Extract DoF ownership information
    lambda_locally_owned_dofs = dof_handler_trace.locally_owned_dofs();
    flux_locally_owned_dofs = dof_handler_flux.locally_owned_dofs();
    head_locally_owned_dofs = dof_handler_head.locally_owned_dofs();

    DoFTools::extract_locally_relevant_dofs(dof_handler_trace,lambda_locally_relevant_dofs);
    DoFTools::extract_locally_relevant_dofs(dof_handler_head,head_locally_relevant_dofs);
    DoFTools::extract_locally_relevant_dofs(dof_handler_flux,flux_locally_relevant_dofs);

    // 4. Build constraints for the trace (unchanged)
    lambda_constraints.clear();
    lambda_constraints.reinit(lambda_locally_relevant_dofs);

    //Apply hanging node constraints
    DoFTools::make_hanging_node_constraints(dof_handler_trace,lambda_constraints);

    // Apply boundary conditions (call the new function)
    apply_trace_boundary_conditions();

    lambda_constraints.close();

    // lambda_ownership knows which ranges each processor owns
    lambda_ownership.reinit(lambda_locally_owned_dofs, mpi_communicator, uo.verbose_level);


    setup_local_cell_well_link();
    setup_well_index_sets_by_segments();
    update_local_cell_well_link_owners();
    build_trace_well_coupling_maps();

    //const unsigned int n_wells  = mnwells.wells.size();

    // ------------------------------
    // (A) Initialize block sparsity maps (owned/owned only)
    // ------------------------------
    TrilinosWrappers::BlockSparsityPattern block_sparsity;
    block_sparsity.reinit(2, 2);

    block_sparsity.block(0,0).reinit(lambda_locally_owned_dofs,
                                     lambda_locally_owned_dofs,
                                     mpi_communicator);

    block_sparsity.block(0,1).reinit(lambda_locally_owned_dofs,
                                     well_locally_owned_dofs,
                                     mpi_communicator);

    block_sparsity.block(1,0).reinit(well_locally_owned_dofs,
                                     lambda_locally_owned_dofs,
                                     mpi_communicator);

    block_sparsity.block(1,1).reinit(well_locally_owned_dofs,
                                     well_locally_owned_dofs,
                                     mpi_communicator);
    block_sparsity.collect_sizes();

    // ------------------------------
    // (B) Fill (0,0) with standard trace connectivity (MPI-safe)
    // ------------------------------
    {
        DynamicSparsityPattern dsp_00(lambda_locally_relevant_dofs);
        DoFTools::make_sparsity_pattern(dof_handler_trace,
                                  block_sparsity.block(0,0),
                                  lambda_constraints,
                                  /*keep_constrained_dofs=*/false);

        SparsityTools::distribute_sparsity_pattern(dsp_00,
                                             lambda_locally_owned_dofs,
                                             mpi_communicator,
                                             lambda_locally_relevant_dofs);

        //block_sparsity.block(0,0).copy_from(dsp_00);
    }

    // ------------------------------
    // (C) Fill (0,1): trace-row couplings to wells
    //     Use local_trace_well_pairs (from locally owned cells)
    //     Only add rows that I own: lambda_locally_owned_dofs
    // ------------------------------
    // After build_trace_well_coupling_maps() has run:

    // block_sparsity.block(0,1) has row space = lambda_locally_owned_dofs
    // and column space = well_locally_owned_dofs? (depends on your block setup)
    // Typically for a distributed block sparsity you add (row,col) even if col is nonlocal,
    // as long as the sparsity object supports it (Trilinos/LA may need specific init).
    // Here we follow the old behavior: add for locally owned trace rows only.
    for (const auto &kv : trace_to_well_dof.data())
    {
        const auto trace_row = kv.first;
        AssertThrow(lambda_locally_owned_dofs.is_element(trace_row), ExcInternalError());

        for (const auto &wr : kv.second)
            block_sparsity.block(0,1).add(trace_row, wr.well_id);
    }

    // ------------------------------
    // (D) Fill (1,0): well-row couplings to trace columns
    //     Use global_trace_well_pairs which (by new gather) contains
    //     only pairs for wells I own.
    // ------------------------------
    for (const auto &kv : well_to_trace_dof.data())
    {
        const unsigned int well_row = static_cast<unsigned int>(kv.first);
        AssertThrow(well_locally_owned_dofs.is_element(well_row), ExcInternalError());

        for (const auto &tr : kv.second)
            block_sparsity.block(1,0).add(well_row, tr.trace_id); // trace_id may be remote
    }

    // ------------------------------
    // (E) Fill (1,1): diagonal well coupling (owned rows only)
    // ------------------------------
    for (auto it = well_locally_owned_dofs.begin(); it != well_locally_owned_dofs.end(); ++it)
    {
        const types::global_dof_index w = *it;      // global well id
        block_sparsity.block(1,1).add(w, w);
    }

    // Finalize pattern
    block_sparsity.compress();

    // Build Trilinos block matrix using block_sparsity
    block_system_matrix.reinit(block_sparsity);

    // ------------------------------
    // (F) Build block vectors
    // ------------------------------
    std::vector<IndexSet> owned(2), relevant(2);
    owned[0]    = lambda_locally_owned_dofs;
    owned[1]    = well_locally_owned_dofs;
    // For vectors, "relevant" may overlap; this is okay.
    relevant[0] = lambda_locally_relevant_dofs;
    relevant[1] = well_locally_relevant_dofs;

    block_rhs_vector.reinit(owned, mpi_communicator);
    block_solution.reinit(owned, relevant, mpi_communicator);

    // 8. Allocate storage for transient head update
    //h_old.reinit(head_locally_owned_dofs, head_locally_relevant_dofs, mpi_communicator);
    h_old.reinit(head_locally_owned_dofs, mpi_communicator);
    //h_new.reinit(head_locally_owned_dofs, head_locally_relevant_dofs, mpi_communicator);
    h_new.reinit(head_locally_owned_dofs, mpi_communicator);
    //h_new_owned.reinit(head_locally_owned_dofs, mpi_communicator);
    q_new.reinit(flux_locally_owned_dofs, flux_locally_relevant_dofs, mpi_communicator);
    well_solution.reinit(well_locally_owned_dofs, mpi_communicator);
    // Separate ghosted read vector (owned + relevant)
    well_solution_ghosted.reinit(well_locally_owned_dofs,
                                 well_locally_relevant_dofs,
                                 mpi_communicator);
    h_guess.reinit(head_locally_owned_dofs,mpi_communicator);

    // 9. Output statistics
    pcout << "Block MNW2-HMFEM system setup complete:" << std::endl;
    pcout << "  λ DOFs = " << dof_handler_trace.n_dofs() << std::endl;
    pcout << "  Well DOFs (global) = " << mnwells.wells.size() << std::endl;
    pcout << "  Well DOFs (owned here) = " << well_locally_owned_dofs.n_elements() << std::endl;
    pcout << "  u DOFs = " << dof_handler_flux.n_dofs() << std::endl;
    pcout << "  h DOFs = " << dof_handler_head.n_dofs() << std::endl;

    save_velocity_io_mapping_once();
}

template <int dim>
void NPSAT_FLOW<dim>::apply_trace_boundary_conditions() {
    pcout << "Applying trace boundary conditions..." << std::endl;

    for (typename decltype(dirichlet_boundary_map)::const_iterator it =
             dirichlet_boundary_map.begin();
         it != dirichlet_boundary_map.end();
         ++it) {
        const types::boundary_id boundary_id = it->first;
        const auto function_ptr = it->second;
        if (function_ptr != nullptr)
        {
            VectorTools::interpolate_boundary_values(
                dof_handler_trace,
                boundary_id,
                *function_ptr,
                lambda_constraints);
        }
    }
}

template <int dim>
void NPSAT_FLOW<dim>::setup_local_cell_well_link() {
    local_cell_well_map.clear();

    const unsigned int n_trace_dofs = fe_trace.n_dofs_per_cell();
    std::vector<types::global_dof_index> trace_dof_indices(n_trace_dofs);

    npsat_flow::Polygon quad;
    std::vector<double> topZ, bottomZ, Xquad, Yquad;

    typename DoFHandler<dim>::active_cell_iterator trace_cell = dof_handler_trace.begin_active();
    for (; trace_cell != dof_handler_trace.end(); ++trace_cell) {
        if (!trace_cell->is_locally_owned())
            continue;

        std::vector<npsat_flow::MNWell*> wells_in_cell;
        std::vector<double> screen_length_inside;
        std::vector<double> well_z_bot;

        npsat_flow::quad_and_Zcoords_from_cell<dim, typename DoFHandler<dim>::active_cell_iterator>(
            quad, trace_cell, Xquad, Yquad, topZ, bottomZ);
        mnwells.find_wells_in_polygon(quad, wells_in_cell,screen_length_inside,well_z_bot,
            Xquad,Yquad,topZ,bottomZ);

        if (wells_in_cell.empty())
            continue;

        double ro, b;
        npsat_flow::calc_ro<dim>(trace_cell, ro, b);
        Point<dim> cell_centroid = trace_cell->center();

        std::vector<npsat_flow::CellWellLink> links;
        trace_cell->get_dof_indices(trace_dof_indices);

        for (unsigned int i_well = 0; i_well < wells_in_cell.size(); ++i_well) {
            const auto *well_ptr = wells_in_cell[i_well];
            const unsigned int well_id = well_ptr->global_index;
            Tensor<2,dim> Kcond = hgeo_prop.conductivity(Point<dim>(well_ptr->x, well_ptr->y, cell_centroid(2)));

            const double KK = std::sqrt(Kcond[0][0]*Kcond[1][1]);
            const double bw = screen_length_inside[i_well];

            //const double cwc2 = npsat_v2::compute_CWC_MNW2(KK, b, bw, well->rw, ro,well->Kskin, well->Rskin );
            const double loss_denom = 2*numbers::PI * b * KK;
            const double ln_ro_rw = std::log(ro/well_ptr->rw);

            /*
             * Interpret well_ptr->Rskin as:
             *   0 < Rskin <= 1 : fraction of element diameter
             *   Rskin > 1      : length units
             *
             * For fraction input, constrain to [0.1, 0.5].
             * Then force the physical skin radius to be larger than rw.
             */

            const double Rskin_fraction_min = 0.1;
            const double Rskin_fraction_max = 0.5;
            const double Rskin_rw_factor    = 1.1;
            const double Rskin_ro_factor    = 0.8;

            double Rskin_length = well_ptr->Rskin;
            if (well_ptr->Rskin > 0.0 && well_ptr->Rskin <= 1.0) {
                double Rskin_fraction = well_ptr->Rskin;
                if (Rskin_fraction < Rskin_fraction_min)
                {
                    std::cout << "Warning: well " << well_id
                          << " has Rskin fraction " << Rskin_fraction
                          << " smaller than " << Rskin_fraction_min
                          << ". Using " << Rskin_fraction_min << "." << std::endl;

                    Rskin_fraction = Rskin_fraction_min;
                }

                if (Rskin_fraction > Rskin_fraction_max)
                {
                    pcout << "Warning: well " << well_id
                          << " has Rskin fraction " << Rskin_fraction
                          << " larger than " << Rskin_fraction_max
                          << ". Using " << Rskin_fraction_max << "." << std::endl;

                    Rskin_fraction = Rskin_fraction_max;
                }

                /*
                 * Approximate element diameter from Peaceman radius:
                 * ro = 0.14 * sqrt(dx^2 + dy^2)
                 * so element_diameter = sqrt(dx^2 + dy^2) = ro / 0.14
                 */
                const double element_diameter = ro / 0.14;
                Rskin_length = Rskin_fraction * element_diameter;
            }

            const double Rskin_min = Rskin_rw_factor * well_ptr->rw;
            const double Rskin_max = Rskin_ro_factor * ro;

            double Rskin_eff = std::max(Rskin_length, Rskin_min);
            if (Rskin_eff > Rskin_max)
            {
                pcout << "Warning: well " << well_id
                      << " has effective Rskin " << Rskin_eff
                      << " larger than " << Rskin_max
                      << " = " << Rskin_ro_factor << "*ro. Using " << Rskin_max
                      << "." << std::endl;

                Rskin_eff = Rskin_max;
            }

            AssertThrow(Rskin_eff > well_ptr->rw, ExcMessage("Invalid Rskin_eff: skin radius must be larger than rw."));

            AssertThrow(Rskin_eff < ro, ExcMessage("Invalid Rskin_eff: skin radius must be smaller than ro."));

            const double skin_min = -0.8 * ln_ro_rw;  // stimulation can reduce but not erase aquifer loss
            double skin = (KK*b / (well_ptr->Kskin * bw) - 1.0)*std::log(Rskin_eff/well_ptr->rw);
            skin = std::max(skin, skin_min);
            double D = ln_ro_rw+ skin;

            const double Dmin = 0.5; // pick 0.2–1.0; 0.5 is a good start
            if (D < Dmin)
                D = Dmin;
            const double cwc = loss_denom / D;

            // pcout << "well " << well->global_index << ", \t"
            //       << "ro " << ro << ", \t"
            //       << "b " << b << ", \t"
            //       << "bw " << bw << ", \t"
            //       << "loss_denom " << loss_denom << ", \t"
            //       << "skin " << skin << ", \t"
            //       << "ln(ro/rw) " << std::log(ro / well->rw) << ", \t"
            //       << "cwc " << cwc << std::endl;


            //pcout << "SL in cell: " << screen_length_inside[i_well] << ", cwc " << cwc << ", cwc2 " << cwc2 << std::endl;
            npsat_flow::CellWellLink link;
            link.well_global_index = well_id;
            link.well_owner_rank = numbers::invalid_unsigned_int;
            link.cwc = cwc;
            link.sl = screen_length_inside[i_well];
            link.w_zbot = well_z_bot[i_well];
            link.ze = cell_centroid(2);
            link.trace_dof_indices.insert( link.trace_dof_indices.end(),
                    trace_dof_indices.begin(), trace_dof_indices.end());
            links.push_back(std::move(link));
        }
        const unsigned int aidx = trace_cell->active_cell_index();
        local_cell_well_map.emplace_back(aidx, std::move(links));
        local_cell_id_strings.emplace_back(aidx, trace_cell->id().to_string());
    }
    // Sort for binary search lookup
    auto key_comp = [](const auto &A, const auto &B){ return A.first < B.first; };
    std::sort(local_cell_well_map.begin(), local_cell_well_map.end(), key_comp );
    std::sort(local_cell_id_strings.begin(), local_cell_id_strings.end(), key_comp);
    std::cout << "Rank: " << my_rank << " has " << local_cell_well_map.size() << " cell-well links" << std::endl;
    if (uo.print_cell_well_map_csv)
        npsat_flow::print_cell_well_map_csv(output_prefix_path(), local_cell_well_map, my_rank);
}


template <int dim>
void NPSAT_FLOW<dim>::setup_well_index_sets_by_segments() {
    const unsigned int n_wells = static_cast<unsigned int>(mnwells.wells.size());

    // ------------------------------------------------------------------
    // (A) Local weight per well (screen length sum or count)
    // ------------------------------------------------------------------
    std::vector<double> local_weight(n_wells, 0.0);
    std::vector<unsigned int> local_touched(n_wells, 0u);
    // Use screen length "sl" as segment weight:
    for (const auto &cell_entry : local_cell_well_map)
    {
        for (const auto &link : cell_entry.second)
        {
            const unsigned int w = link.well_global_index;
            AssertThrow(w < n_wells, ExcMessage("CellWellLink.well_global_index out of range"));
            // By screen length
            //local_weight[w] += link.sl;
            //By count
            local_weight[w] += 1.0;
            local_touched[w] = 1u;
        }
    }

    // ------------------------------------------------------------------
    // (B) Check for zero cell wells
    // ------------------------------------------------------------------
    {
        std::vector<unsigned int> global_touched(n_wells, 0u);
        const int ierr = MPI_Allreduce(local_touched.data(),
                                       global_touched.data(),
                                       static_cast<int>(n_wells),
                                       MPI_UNSIGNED,
                                       MPI_SUM,
                                       mpi_communicator);
        AssertThrow(ierr == MPI_SUCCESS, ExcMessage("MPI_Allreduce(touched) failed"));

        std::vector<unsigned int> zero_wells;
        for (unsigned int w = 0; w < n_wells; ++w)
            if (global_touched[w] == 0u)
                zero_wells.push_back(w);

        if (!zero_wells.empty())
        {
            if (my_rank == 0)
            {
                pcout << "ERROR: Found " << zero_wells.size()
                      << " wells with zero intersections (likely outside domain).\n"
                      << "First well IDs (up to 50): ";
                const unsigned int cap = std::min<unsigned int>(50, zero_wells.size());
                for (unsigned int k = 0; k < cap; ++k)
                {
                    if (k) pcout << ",";
                    pcout << zero_wells[k];
                }
                pcout << std::endl;
            }
            AssertThrow(zero_wells.empty(),
                        ExcMessage("Wells with zero intersections detected."));
        }
    }

    // ------------------------------------------------------------------
    // (C) Argmax across ranks using the STANDARD MPI pair type
    // MPI_DOUBLE_INT corresponds to struct { double val; int rank; } for MAXLOC/MINLOC.
    // ------------------------------------------------------------------
    std::vector<std::pair<double,int>> local_pair(n_wells), global_pair(n_wells);

    for (unsigned int w = 0; w < n_wells; ++w)
    {
        local_pair[w] = { local_weight[w], static_cast<int>(my_rank) };
    }

    {
        const int ierr = MPI_Allreduce(local_pair.data(),
                                       global_pair.data(),
                                       static_cast<int>(n_wells),
                                       MPI_DOUBLE_INT,
                                       MPI_MAXLOC,
                                       mpi_communicator);
        AssertThrow(ierr == MPI_SUCCESS, ExcMessage("MPI_Allreduce(MPI_MAXLOC) with MPI_DOUBLE_INT failed"));
    }

    // ------------------------------------------------------------------
    // (D) Build owner mapping
    // ------------------------------------------------------------------
    well_owner_rank.assign(n_wells, 0u);
    for (unsigned int w = 0; w < n_wells; ++w)
        well_owner_rank[w] = static_cast<unsigned int>(global_pair[w].second);

    // ------------------------------------------------------------------
    // (E) Build owned IndexSet
    // ------------------------------------------------------------------
    well_locally_owned_dofs.clear();
    well_locally_owned_dofs.set_size(n_wells);

    for (unsigned int w = 0; w < n_wells; ++w)
    {
        if (well_owner_rank[w] == my_rank)
            well_locally_owned_dofs.add_index(w);
    }

    well_locally_owned_dofs.compress();

    // relevant is used only for reading well head results not for sparsity
    well_locally_relevant_dofs.clear();
    well_locally_relevant_dofs.set_size(n_wells);
    well_locally_relevant_dofs.add_range(0, n_wells);
    well_locally_relevant_dofs.compress();

    // ------------------------------------------------------------------
    // (F) Diagnostics
    // ------------------------------------------------------------------
    const unsigned int owned_local = well_locally_owned_dofs.n_elements();
    const unsigned int owned_sum   = Utilities::MPI::sum(owned_local, mpi_communicator);
    const unsigned int owned_min   = Utilities::MPI::min(owned_local, mpi_communicator);
    const unsigned int owned_max   = Utilities::MPI::max(owned_local, mpi_communicator);

    pcout << "Well DoFs (by segments): n_wells=" << n_wells
              << ", sum(owned)=" << owned_sum << " (should equal n_wells)"
              << ", per-rank min=" << owned_min
              << ", max=" << owned_max
              << std::endl;

    AssertThrow(owned_sum == n_wells,
                ExcMessage("Well DoF ownership is not a valid partition."));

    { //TODO THis should be removed in real applications
        for (unsigned int i = 0; i < n_proc; ++i)
        {
            // Ensure ranks print in order 0,1,2,... with no interleaving
            MPI_Barrier(mpi_communicator);

            if (i == my_rank)
            {
                std::ostringstream oss;
                oss << "Rank " << my_rank << " owns well DoFs: ";

                bool first = true;
                for (auto it = well_locally_owned_dofs.begin();
                     it != well_locally_owned_dofs.end(); ++it)
                {
                    if (!first) oss << ",";
                    oss << static_cast<types::global_dof_index>(*it);
                    first = false;
                }

                std::cout << oss.str() << std::endl << std::flush;
            }

            MPI_Barrier(mpi_communicator);
        }
    }
}

template <int dim>
void NPSAT_FLOW<dim>::update_local_cell_well_link_owners() {
    for (auto &cell_entry : local_cell_well_map)
        for (auto &link : cell_entry.second)
            link.well_owner_rank = well_owner_rank[link.well_global_index];
}

template <int dim>
void NPSAT_FLOW<dim>::build_trace_well_coupling_maps() {
    // ---- temporary accumulators (one entry per key)
    std::unordered_map<types::global_dof_index, std::vector<npsat_flow::WellRef>>  tr2w_tmp;
    std::unordered_map<unsigned int,            std::vector<npsat_flow::TraceRef>> w2tr_tmp;

    // ---- MPI send buffers
    std::vector<std::vector<npsat_flow::TraceWellEdgeMsg>> send_to_trace_owner(n_proc);
    std::vector<std::vector<npsat_flow::WellTraceEdgeMsg>> send_to_well_owner(n_proc);

    std::vector<types::global_dof_index> expanded;
    expanded.reserve(16);

    auto trace_owner_of = [&](const types::global_dof_index tr) -> unsigned int
    {
        if (lambda_locally_owned_dofs.is_element(tr))
            return my_rank;
        return lambda_ownership.get_owner(tr); // range lookup, no cache
    };

    // ============================================================================
    // (1) Local discovery from local_cell_well_map and routing to owners
    // ============================================================================
    for (const auto &cell_entry : local_cell_well_map)
    {
        const auto &links = cell_entry.second;

        for (const auto &link : links)
        {
            const unsigned int well_id   = link.well_global_index;
            const unsigned int well_rank = link.well_owner_rank; // already filled

            for (const auto tr_raw : link.trace_dof_indices)
            {
                npsat_flow::expand_trace_by_constraints(tr_raw, lambda_constraints, expanded);
                if (expanded.empty())
                    continue;

                for (const auto tr : expanded)
                {
                    const unsigned int tr_rank = trace_owner_of(tr);

                    // (A) trace owner gets trace->well adjacency (keys are local traces there)
                    if (tr_rank == my_rank)
                    {
                        // store only if this trace is locally owned (should be, since tr_rank==my_rank)
                        AssertThrow(lambda_locally_owned_dofs.is_element(tr),
                            ExcMessage("trace_owner_of() returned my_rank but trace is not locally owned."));
                        tr2w_tmp[tr].push_back({well_id, well_rank});
                    }
                    else
                    {
                        send_to_trace_owner[tr_rank].push_back({tr, well_id, well_rank});
                    }

                    // (B) well owner gets well->trace adjacency (keys are local wells there)
                    if (well_rank == my_rank)
                    {
                        AssertThrow(well_locally_owned_dofs.is_element(well_id),
                            ExcMessage("well_rank == my_rank but well is not locally owned."));
                        w2tr_tmp[well_id].push_back({tr, tr_rank});
                    }
                    else
                    {
                        send_to_well_owner[well_rank].push_back({well_id, tr, tr_rank});
                    }
                }
            }
        }
    }

    // ============================================================================
    // (2) MPI exchange
    // ============================================================================
    const auto recv_for_traces =
    npsat_flow::all_to_all_vector_of_vectors<npsat_flow::TraceWellEdgeMsg>(mpi_communicator,
                                                          send_to_trace_owner);
    const auto recv_for_wells =
    npsat_flow::all_to_all_vector_of_vectors<npsat_flow::WellTraceEdgeMsg>(mpi_communicator,
                                                          send_to_well_owner);

    // Integrate received edges for locally owned traces (trace_to_well_dof)
    for (const auto &bucket : recv_for_traces)
        for (const auto &m : bucket)
        {
            AssertThrow(lambda_locally_owned_dofs.is_element(m.trace_id),
                        ExcMessage("Received trace edge for trace not owned on this rank."));
            tr2w_tmp[m.trace_id].push_back({m.well_id, m.well_rank});
        }

    // Integrate received edges for locally owned wells (well_to_trace_dof)
    for (const auto &bucket : recv_for_wells)
        for (const auto &m : bucket)
        {
            AssertThrow(well_locally_owned_dofs.is_element(m.well_id),
                        ExcMessage("Received well edge for well not owned on this rank."));
            w2tr_tmp[m.well_id].push_back({m.trace_id, m.trace_rank});
        }

    // ============================================================================
    // (3) Sort+unique per key to remove duplicates
    // ============================================================================
    auto unique_wellrefs = [](std::vector<npsat_flow::WellRef> &v)
    {
        std::sort(v.begin(), v.end(),
                  [](const auto &a, const auto &b)
                  { return (a.well_id < b.well_id) || (a.well_id == b.well_id && a.well_rank < b.well_rank); });
        v.erase(std::unique(v.begin(), v.end(),
                            [](const auto &a, const auto &b) { return a.well_id == b.well_id; }),
                v.end());
    };

    auto unique_tracerefs = [](std::vector<npsat_flow::TraceRef> &v)
    {
        std::sort(v.begin(), v.end(),
                  [](const auto &a, const auto &b)
                  { return (a.trace_id < b.trace_id) || (a.trace_id == b.trace_id && a.trace_rank < b.trace_rank); });
        v.erase(std::unique(v.begin(), v.end(),
                            [](const auto &a, const auto &b) { return a.trace_id == b.trace_id; }),
                v.end());
    };

    for (auto &kv : tr2w_tmp) unique_wellrefs(kv.second);
    for (auto &kv : w2tr_tmp) unique_tracerefs(kv.second);

    // ============================================================================
    // (4) Populate SortedVectorMap (ONE insert per key), then sort
    //     NOTE: This requires SortedVectorMap<Key,T> keyed by global dof id type.
    // ============================================================================
    trace_to_well_dof = decltype(trace_to_well_dof)(tr2w_tmp.size());
    for (auto &kv : tr2w_tmp)
        trace_to_well_dof.insert(kv.first, kv.second);
    trace_to_well_dof.sort();

    well_to_trace_dof = decltype(well_to_trace_dof)(w2tr_tmp.size());
    for (auto &kv : w2tr_tmp)
        well_to_trace_dof.insert(kv.first, kv.second); // ideally key is unsigned/global type
    well_to_trace_dof.sort();

    if (uo.print_trace_well_maps_csv)
    {
        const std::string prefix = output_prefix_path();
        trace_to_well_dof.print_data(prefix, my_rank);
        well_to_trace_dof.print_data(prefix, my_rank);
    }
}

template <int dim>
void NPSAT_FLOW<dim>::initialize_initial_head() {
    AssertThrow(!uo.initial_head_file.empty(),
                ExcMessage("IC.Head must specify an initial-head interpolation master file."));
    AssertThrow(h_old.size() == dof_handler_head.n_dofs(),
                ExcMessage("Head vector is not initialized. Call setup_system() before initialize_initial_head()."));

    const std::string input_root = npsat_flow::join_paths(uo.main_path, uo.input_path);
    auto head_interp = std::make_shared<npsat_flow::InterpInterface<dim>>();

    head_interp->read_master_file(uo.initial_head_file,
                                  1.0,
                                  mpi_communicator,
                                  input_root);

    npsat_flow::InterpolationFunction<dim> initial_head;
    initial_head.set_interpolant(head_interp);
    initial_head.set_time_index(0);

    VectorTools::interpolate(dof_handler_head, initial_head, h_old);
    h_old.compress(VectorOperation::insert);

    pcout << "Initial head loaded from "
          << npsat_flow::resolve_relative_path(input_root, uo.initial_head_file)
          << std::endl;
}

#endif //NPSAT_FLOW_SETUP_IMPL_H
