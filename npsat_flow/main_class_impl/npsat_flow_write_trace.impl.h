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

#endif //NPSAT_FLOW_WRITE_TRACE_IMPL_H
