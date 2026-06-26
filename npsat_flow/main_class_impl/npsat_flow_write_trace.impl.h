//
// Created by giorgk on 6/25/26.
//

#ifndef NPSAT_FLOW_WRITE_TRACE_IMPL_H
#define NPSAT_FLOW_WRITE_TRACE_IMPL_H

template <int dim>
void NPSAT_FLOW<dim>::save_velocity_io_mapping_once() const {
    if (!uo.save_trace_data || time_tracking.simulation_step() != 0)
        return;

    AssertThrow(fe_flux.n_dofs_per_face() == 1, ExcMessage("Expected RT0-like: 1 DoF per face."));

    const unsigned int faces_per_cell = GeometryInfo<dim>::faces_per_cell;

    const std::string str_rank = Utilities::int_to_string(my_rank, 4);
    const std::string prefix = output_prefix_path();
    const std::string filename = prefix + "_velmap_rank_" + str_rank + ".bin";

    pcout << "Saving velocity IO mapping (once): " << filename << std::endl;

    std::ofstream out(filename, std::ios::binary);
    AssertThrow(out.good(), ExcMessage("Could not open file for writing: " + filename));

    // ---------------- Header ----------------
    const std::uint64_t magic   = 0x4E50534154564D50ull; // "NPSATVMP" (arbitrary)
    const std::uint32_t version = 2;
    const std::uint32_t dim_u32 = dim;
    const std::uint32_t fpc_u32 = faces_per_cell;

    const std::uint64_t n_global_dofs = dof_handler_flux.n_dofs();
    const std::uint64_t n_owned_dofs  = flux_locally_owned_dofs.n_elements();
    const std::uint64_t n_local_cells = triangulation.n_locally_owned_active_cells();

    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&dim_u32), sizeof(dim_u32));
    out.write(reinterpret_cast<const char*>(&fpc_u32), sizeof(fpc_u32));
    out.write(reinterpret_cast<const char*>(&n_global_dofs), sizeof(n_global_dofs));
    out.write(reinterpret_cast<const char*>(&n_owned_dofs), sizeof(n_owned_dofs));
    out.write(reinterpret_cast<const char*>(&n_local_cells), sizeof(n_local_cells));

    // ---------------- Section A: owned DoF gid order ----------------
    // This must match your per-step writing loop order exactly.
    for (auto it = flux_locally_owned_dofs.begin(); it != flux_locally_owned_dofs.end(); ++it)
    {
        const std::uint64_t gid64 = static_cast<std::uint64_t>(*it);
        out.write(reinterpret_cast<const char*>(&gid64), sizeof(gid64));
    }

    // For writing the mapping, we want for each cell face:
    // sign = +1 if THIS cell is the canonical owner (writer) for that face_gid, else -1.
    auto sign_to_outward_for_cell = [&](const typename DoFHandler<dim>::active_cell_iterator &cell,
                                            const unsigned int f) -> std::int8_t
    {
        if (cell->at_boundary(f))
            return static_cast<std::int8_t>(+1);

        const auto neigh = cell->neighbor(f);

        // Coarse-side parent face at coarse-fine interface:
        // stored parent-face value is outward w.r.t. this coarse cell.
        if (!neigh->is_active())
            return static_cast<std::int8_t>(+1);

        // Refined-side active child face at coarse-fine interface:
        // stored child-face value is outward w.r.t. this refined child cell.
        if (neigh->level() < cell->level())
            return static_cast<std::int8_t>(+1);

        // Same-level shared face:
        // stored value is outward w.r.t. the deterministic owner cell.
        return (cell->id() < neigh->id())
                 ? static_cast<std::int8_t>(+1)
                 : static_cast<std::int8_t>(-1);
    };

    // ---------------- Section B: per-cell face gid + sign ----------------
    // Deterministic cell order: sort by CellId string
    struct CellRec
    {
        std::string cid;
        std::array<types::global_dof_index, GeometryInfo<dim>::faces_per_cell> face_gid;
        std::array<std::int8_t, GeometryInfo<dim>::faces_per_cell>                   sign;
        std::array<std::uint8_t, GeometryInfo<dim>::faces_per_cell>                  flags;
    };

    std::vector<CellRec> recs;
    recs.reserve(static_cast<std::size_t>(n_local_cells));

    std::vector<types::global_dof_index> face_flux_dofs(fe_flux.n_dofs_per_face());
    AssertThrow(face_flux_dofs.size() == 1, ExcMessage("RT0: expected 1 dof per face."));

    for (const auto &cell : dof_handler_flux.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        CellRec r;
        r.cid = cell->id().to_string();

        for (unsigned int f = 0; f < faces_per_cell; ++f)
        {
            cell->face(f)->get_dof_indices(face_flux_dofs, 0);
            r.face_gid[f] = face_flux_dofs[0];
            r.sign[f]     = sign_to_outward_for_cell(cell, f);

            std::uint8_t fl = 0;
            if (!cell->at_boundary(f))
            {
                const auto neigh = cell->neighbor(f);
                if (!neigh->is_active())
                    fl |= 0x1; // coarse-side parent face
                else if (neigh->level() < cell->level())
                    fl |= 0x2; // refined-side active child face
            }
            r.flags[f] = fl;
        }
        recs.emplace_back(std::move(r));
    }

    std::sort(recs.begin(), recs.end(),
              [](const CellRec &a, const CellRec &b){ return a.cid < b.cid; });

    for (const auto &r : recs)
    {
        const std::uint16_t len = static_cast<std::uint16_t>(r.cid.size());
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(r.cid.data(), static_cast<std::streamsize>(len));

        for (unsigned int f = 0; f < faces_per_cell; ++f)
        {
            const std::uint64_t gid64 = static_cast<std::uint64_t>(r.face_gid[f]);
            out.write(reinterpret_cast<const char*>(&gid64), sizeof(gid64));
            out.write(reinterpret_cast<const char*>(&r.sign[f]), sizeof(std::int8_t));
            out.write(reinterpret_cast<const char*>(&r.flags[f]), sizeof(std::uint8_t));
        }
    }

    AssertThrow(out.good(), ExcMessage("Failed while writing mapping file: " + filename));
    out.close();
    MPI_Barrier(mpi_communicator);
}

template<int dim>
void NPSAT_FLOW<dim>::export_cell_well_map_binary_once(const std::string &prefix) const {
    if (!uo.save_trace_data || time_tracking.simulation_step() != 0)
        return;
    pcout << "\t export_cell_well_map_binary_once..." <<std::endl;

    const std::string rank_s = dealii::Utilities::int_to_string(my_rank, 4);
    const std::string fname  = prefix + "_cellwell_map_rank_" + rank_s + ".bin";

    std::ofstream out(fname, std::ios::binary);
    AssertThrow(out.good(), dealii::ExcMessage("Could not open: " + fname));

    // header
    const char magic[8] = {'C','W','M','A','P','v','2','\0'};
    out.write(magic, 8);
    const std::uint32_t version = 2;
    const std::uint32_t dim_u32 = dim;
    npsat_flow::write_pod(out, version);
    npsat_flow::write_pod(out, dim_u32);

    const std::uint64_t n_cells = static_cast<std::uint64_t>(local_cell_well_map.size());
    npsat_flow::write_pod(out, n_cells);

    for (const auto &cell_pair : local_cell_well_map) {
        const auto active_idx = cell_pair.first;
        const auto &links     = cell_pair.second;

        const std::string cell_id_str = cellid_string_from_active_index(active_idx);
        npsat_flow::write_string(out, cell_id_str);

        const std::uint32_t n_links = static_cast<std::uint32_t>(links.size());
        npsat_flow::write_pod(out, n_links);

        for (const auto &L : links) {
            const std::uint32_t wgid = static_cast<std::uint32_t>(L.well_global_index);
            npsat_flow::write_pod(out, wgid);

            // Keep owner rank (even if you might recompute later)
            npsat_flow::write_pod(out, static_cast<std::uint16_t>(L.well_owner_rank));

            // Pull stable well metadata (XY etc) from mnwells
            AssertThrow(wgid < mnwells.wells.size(),
                        dealii::ExcMessage("Invalid well_global_index in local_cell_well_map."));
            const auto &W = mnwells.wells[wgid];

            const std::int32_t eid  = static_cast<std::int32_t>(W.Eid);
            const double wx = W.x;
            const double wy = W.y;
            const double wtop = W.top;
            const double wbot = W.bottom;
            const std::int32_t q_row = static_cast<std::int32_t>(W.q_row);

            npsat_flow::write_pod(out, eid);
            npsat_flow::write_pod(out, wx);
            npsat_flow::write_pod(out, wy);
            npsat_flow::write_pod(out, wtop);
            npsat_flow::write_pod(out, wbot);
            npsat_flow::write_pod(out, q_row);

            // Your existing link geometry fields (stay; used by tracer)
            npsat_flow::write_pod(out, L.ze);
            npsat_flow::write_pod(out, L.sl);
            npsat_flow::write_pod(out, L.w_zbot);
        }
    }
    AssertThrow(out.good(), dealii::ExcMessage("Write failed: " + fname));
}

template<int dim>
void NPSAT_FLOW<dim>::save_water_table_per_step(const std::string &prefix) const {
    pcout << "\t save_water_table_per_step..." << std::endl;

    const unsigned int step_no = time_tracking.simulation_step();
    const std::string step_s   = Utilities::int_to_string(step_no, 3);
    const std::string rank_s   = Utilities::int_to_string(my_rank, 4);

    struct WaterTableRecord
    {
        std::string cell_id;
        double x = 0.0;
        double y = 0.0;
        double water_table_elevation = 0.0;
        double top_cell_elevation = 0.0;
        double effective_top_elevation = 0.0;
        bool uses_effective_top = false;
    };

    std::vector<WaterTableRecord> ascii_records;
    ascii_records.reserve(1024);

    std::vector<WaterTableRecord> binary_records;
    binary_records.reserve(1024);

    const unsigned int n_head_dofs = fe_head.n_dofs_per_cell();
    std::vector<types::global_dof_index> head_dof_indices(n_head_dofs);

    const double tol = 1e-12;
    for (auto head_cell = dof_handler_head.begin_active(); head_cell != dof_handler_head.end(); ++head_cell) {
        if (!head_cell->is_locally_owned())
            continue;

        head_cell->get_dof_indices(head_dof_indices);
        AssertThrow(head_dof_indices.size() >= 1, ExcInternalError());

        const double h = h_new[head_dof_indices[0]];

        double z_bot = 0.0;
        double z_top = 0.0;
        const auto bot_face = head_cell->face(4);
        const auto top_face = head_cell->face(5);
        for (unsigned int fv = 0; fv < GeometryInfo<dim>::vertices_per_face; ++fv)
        {
            z_bot += bot_face->vertex(fv)[dim-1];
            z_top += top_face->vertex(fv)[dim-1];
        }
        z_bot /= static_cast<double>(GeometryInfo<dim>::vertices_per_face);
        z_top /= static_cast<double>(GeometryInfo<dim>::vertices_per_face);

        const bool is_partially_saturated = (h > z_bot + tol && h < z_top - tol);
        const bool is_fully_saturated_top_cell = (h >= z_top - tol && head_cell->at_boundary(5));
        const std::uint64_t cell_gid =
            static_cast<std::uint64_t>(head_cell->global_active_cell_index());
        const bool was_recharge_receiver =
            previous_recharge_receiver_gids.find(cell_gid) != previous_recharge_receiver_gids.end();
        const bool effective_top_enabled =
            uo.NLC.effective_top_mode != npsat_flow::NonlinearControls::EffectiveTopMode::Off;
        const bool all_water_table_effective_top =
            uo.NLC.effective_top_mode ==
            npsat_flow::NonlinearControls::EffectiveTopMode::AllWaterTableCells;
        const bool is_effective_top_water_table_cell =
            effective_top_enabled &&
            h > z_bot + tol &&
            (was_recharge_receiver ||
             (all_water_table_effective_top &&
              (h < z_top - tol || head_cell->at_boundary(5))));

        if (!is_partially_saturated &&
            !is_fully_saturated_top_cell &&
            !is_effective_top_water_table_cell)
            continue;

        WaterTableRecord rec;
        rec.cell_id = head_cell->id().to_string();
        rec.x = head_cell->center()[0];
        rec.y = head_cell->center()[1];
        rec.water_table_elevation = h;
        rec.top_cell_elevation = z_top;
        rec.effective_top_elevation = is_effective_top_water_table_cell ? h : z_top;
        rec.uses_effective_top = is_effective_top_water_table_cell;

        ascii_records.push_back(rec);
        if (is_partially_saturated || is_effective_top_water_table_cell)
            binary_records.push_back(std::move(rec));
    }

    if (uo.print_water_table)
    {
        std::ostringstream txt_name;
        txt_name << prefix << "_water_table_rank_" << rank_s
                 << "_step_" << step_s << ".dat";

        std::ofstream out(txt_name.str());
        AssertThrow(out.good(), ExcMessage("Could not open: " + txt_name.str()));

        out << "# x y top_cell effective_top wt uses_effective_top\n";
        out << std::setprecision(16) << std::scientific;
        for (const auto &rec : ascii_records)
        {
            out << rec.x << ' '
                << rec.y << ' '
                << rec.top_cell_elevation << ' '
                << rec.effective_top_elevation << ' '
                << rec.water_table_elevation << ' '
                << rec.uses_effective_top << '\n';
        }
    }

    if (uo.save_trace_data)
    {
        std::ostringstream bin_name;
        bin_name << prefix << "_water_table_rank_" << rank_s
                 << "_step_" << step_s << ".bin";

        std::ofstream out(bin_name.str(), std::ios::binary);
        AssertThrow(out.good(), ExcMessage("Could not open: " + bin_name.str()));

        const char magic[8] = {'W','T','A','B','L','E','v','2'};
        out.write(magic, 8);

        const std::uint32_t version = 2;
        const std::uint32_t dim_u32 = dim;
        const std::uint32_t step_u32 = static_cast<std::uint32_t>(step_no);
        const std::uint64_t nrec = static_cast<std::uint64_t>(binary_records.size());

        npsat_flow::write_pod(out, version);
        npsat_flow::write_pod(out, dim_u32);
        npsat_flow::write_pod(out, step_u32);
        npsat_flow::write_pod(out, nrec);

        for (const auto &rec : binary_records)
        {
            npsat_flow::write_string(out, rec.cell_id);
            npsat_flow::write_pod(out, rec.water_table_elevation);
        }

        AssertThrow(out.good(), ExcMessage("Write failed: " + bin_name.str()));
    }
}

template <int dim>
void NPSAT_FLOW<dim>::save_velocity_per_step(const std::string &prefix) const {
    if (!uo.save_trace_data)
        return;
    pcout << "\t save_velocity_per_step..." << std::endl;
    TrilinosWrappers::MPI::Vector vface_global;
    build_and_write_vface_rt0_per_step(prefix, vface_global);
}

template <int dim>
void NPSAT_FLOW<dim>::build_and_write_vface_rt0_per_step(const std::string &prefix,
    TrilinosWrappers::MPI::Vector &vface_global) const {

    if (!uo.save_trace_data)
        return;

    const unsigned int step_no = time_tracking.simulation_step();
    const std::string step     = Utilities::int_to_string(step_no, 3);
    const std::string str_rank = Utilities::int_to_string(my_rank, 4);

    // Build global RT0 face-normal velocity vector in flux DoF numbering.
    // Stored quantity:
    //   v_n = (1 / |F|) \int_F q_h \cdot n_cell \, dS
    // with sign outward relative to the WRITING cell.
    vface_global.reinit(flux_locally_owned_dofs,  mpi_communicator);//flux_locally_relevant_dofs,
    vface_global = 0.0;

    // Face quadrature and FEFaceValues (same as in output_results)
    QGauss<dim-1> face_quad(fe_flux.degree + 2);
    FEFaceValues<dim> fe_face(fe_flux, face_quad,
                              update_values | update_normal_vectors | update_JxW_values);
    FEValuesExtractors::Vector flux(0);

    const unsigned int n_flux_dofs = fe_flux.n_dofs_per_cell();
    std::vector<types::global_dof_index> flux_dof_indices(n_flux_dofs);
    Vector<double> q_coeff(n_flux_dofs);

    // RT0: 1 DoF per face (we’ll fetch the face DoF id explicitly)
    std::vector<types::global_dof_index> face_flux_dofs(fe_flux.n_dofs_per_face());
    AssertThrow(face_flux_dofs.size() == 1,
                ExcMessage("Expected RT0: fe_flux.n_dofs_per_face() == 1."));

    // Helper: decide face ownership (same-level) + coarse-fine responsibility
    // Ownership rule for tracing export:
    // - boundary face: write it
    // - same-level interior: one deterministic owner writes it
    // - coarse-fine interface: ONLY refined side writes active child-face DoFs
    auto should_write_face =
    [&](const typename DoFHandler<dim>::active_cell_iterator &cell,
        const unsigned int f) -> bool
    {
        if (cell->at_boundary(f))
            return true;

        const auto neigh = cell->neighbor(f);

        if (!neigh->is_active())
        {
            // Neighbor is refined => I am the coarse side.
            // Do NOT write the parent face here. We want subface values later.
            // Let refined children write their own child-face DoFs.
            return false;
        }

        if (neigh->level() < cell->level())
        {
            // Neighbor is coarser => I am refined.
            // I MUST write my child-face DoF, otherwise it stays zero.
            return true;
        }

        // Same level: deterministic owner
        return (cell->id() < neigh->id());
    };

    // Helper: compute outward flux Q = ∫ q_h · n dS on a given cell face
    auto compute_face_flux_outward =
    [&](const typename DoFHandler<dim>::active_cell_iterator &cell,
        const unsigned int f,
        const Vector<double> &q_coeff_cell) -> double
    {
        fe_face.reinit(cell, f);

        double Q_outward = 0.0;
        for (unsigned int q = 0; q < face_quad.size(); ++q)
        {
            Tensor<1, dim> qh;
            for (unsigned int i = 0; i < n_flux_dofs; ++i)
                qh += q_coeff_cell(i) * fe_face[flux].value(i, q);

            const Tensor<1, dim> n = fe_face.normal_vector(q);
            Q_outward += (qh * n) * fe_face.JxW(q);
        }
        return Q_outward;
    };

    // Build vface_global on owned face DoFs
    for (const auto &flux_cell : dof_handler_flux.active_cell_iterators()) {
        if (!flux_cell->is_locally_owned())
            continue;

        // Pull q_h coefficients for this cell (from q_new)
        flux_cell->get_dof_indices(flux_dof_indices);
        for (unsigned int i = 0; i < n_flux_dofs; ++i) {
            q_coeff(i) = q_new[flux_dof_indices[i]];
        }

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
            // (A) Normal path: same-level + refined-side + boundary
            if (!should_write_face(flux_cell, f))
                continue;

            // Determine face DoF id (global). RT0: one per face.
            flux_cell->face(f)->get_dof_indices(face_flux_dofs);
            const types::global_dof_index face_gid = face_flux_dofs[0];

            // Only the owner of this DoF writes to qface_global[face_gid]
            // (prevents illegal writes and avoids races across ranks).
            if (!flux_locally_owned_dofs.is_element(face_gid))
                continue;

            // -----------------------------
            // Compute Q_outward = ∫ q_h · n_cell dS  (cell outward normal)
            // -----------------------------
            // Compute outward flux Q (cell outward normal)
            const double Q_outward = compute_face_flux_outward(flux_cell, f, q_coeff);
            // Convert to NORMAL VELOCITY (flux density)
            const double Af = flux_cell->face(f)->measure();
            AssertThrow(Af > 0.0, ExcMessage("Face area/measure is zero."));
            const double v_outward = Q_outward / Af;
            // Store v on RT0 face DoF index
            vface_global[face_gid] = v_outward;
        }
    }
    // Finalize inserts
    vface_global.compress(VectorOperation::insert);
    vface_global.update_ghost_values();

    // -------------------------
    // Gather values in fixed order (same order as DoF ids file at step 0)
    // -------------------------
    const std::uint64_t n_local = flux_locally_owned_dofs.n_elements();
    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(n_local));

    long double checksum_sum = 0.0L;
    double checksum_maxabs   = 0.0;

    for (auto it = flux_locally_owned_dofs.begin(); it != flux_locally_owned_dofs.end(); ++it)
    {
        const auto gid = *it;
        const double v = vface_global[gid];
        values.push_back(v);
        checksum_sum += static_cast<long double>(v);
        checksum_maxabs = std::max(checksum_maxabs, std::abs(v));
    }

    // -------------------------
    // (A) Binary output (always available; fastest)
    // -------------------------
    bool write_exchange_binary = true;
    if (write_exchange_binary)
    {
        const std::string filename =
            prefix + "_Vface_rt0_vals_rank_" + str_rank + "_step_" + step + ".bin";

        std::ofstream out(filename, std::ios::binary);
        AssertThrow(out.good(), ExcMessage("Could not open file for writing: " + filename));

        // Header:
        // [uint64 magic] [uint32 version]
        // [uint64 step_no] [double time] [double dt]
        // [uint64 n_local]
        // [double checksum_sum] [double checksum_maxabs]
        // [n_local * double values]
        const std::uint64_t magic   = 0x4E505341545F5646ULL; // "NPSAT_VF" (arbitrary)
        const std::uint32_t version = 2;

        const std::uint64_t step_u64 = static_cast<std::uint64_t>(step_no);
        const double t  = time_tracking.simulation_step();
        const double dt = time_tracking.duration();

        const double sum_d    = static_cast<double>(checksum_sum);
        const double maxabs_d = checksum_maxabs;

        out.write(reinterpret_cast<const char*>(&magic),    sizeof(magic));
        out.write(reinterpret_cast<const char*>(&version),  sizeof(version));
        out.write(reinterpret_cast<const char*>(&step_u64), sizeof(step_u64));
        out.write(reinterpret_cast<const char*>(&t),        sizeof(t));
        out.write(reinterpret_cast<const char*>(&dt),       sizeof(dt));
        out.write(reinterpret_cast<const char*>(&n_local),  sizeof(n_local));
        out.write(reinterpret_cast<const char*>(&sum_d),    sizeof(sum_d));
        out.write(reinterpret_cast<const char*>(&maxabs_d), sizeof(maxabs_d));

        if (n_local > 0)
            out.write(reinterpret_cast<const char*>(values.data()),
                      static_cast<std::streamsize>(values.size() * sizeof(double)));

        out.close();
    }

    // -------------------------
    // (B) Optional HDF5 output (only if deal.II built with HDF5)
    // -------------------------
    // bool write_exchange_hdf5 = true;
    // if (write_exchange_hdf5)
    // {
    //     const std::string filename =
    //         prefix + "_Qface_rt0_vals_rank_" + str_rank + "_step_" + step + ".h5";
    //     HDF5::File file(filename, HDF5::File::FileAccessMode::create);
    //
    //     file.write_dataset<double>("/Qface_values", values);
    //
    //     file.set_attribute<std::uint64_t>("/Qface_values", "step", static_cast<std::uint64_t>(step_no));
    //     file.set_attribute<double>("/Qface_values", "time", time_tracking.get_current_time());
    //     file.set_attribute<double>("/Qface_values", "dt",   delta_time);
    //     file.set_attribute<double>("/Qface_values", "checksum_sum", static_cast<double>(checksum_sum));
    //     file.set_attribute<double>("/Qface_values", "checksum_maxabs", checksum_maxabs);
    //
    // }
}

#endif //NPSAT_FLOW_WRITE_TRACE_IMPL_H
