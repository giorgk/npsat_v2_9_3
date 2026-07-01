//
// Created by giorgk on 6/27/26.
//

#ifndef NPSAT_TRACE_LOAD_IMPL_H
#define NPSAT_TRACE_LOAD_IMPL_H

template <int dim>
void NPSAT_TRACE<dim>::load_triangulation() {

    read_parallel_coarse_tria_from_files();

    const std::string dist_base = topt.input_prefix + "_dist_tria";
    const std::string dbg_base  = topt.output_prefix + "_dbg_reloaded";
    // The files expected here are:
    //   dist_base
    //   dist_base.info
    //
    // For example:
    //   prefix_dist_refined
    //   prefix_dist_refined.info

    MPI_Barrier(mpi_communicator);

    // Collective operation on all MPI ranks.
    std::cout << dist_base << std::endl;
    triangulation.load(dist_base.c_str(), false);
    MPI_Barrier(mpi_communicator);

    pcout << "Loaded parallel triangulation: "
         << dist_base
         << std::endl;
         // << "Number of locally owned active cells: "
         // << triangulation.n_locally_owned_active_cells()
         // << std::endl
         // << "Number of global active cells: "
         // << triangulation.n_global_active_cells()
         // << std::endl;
    MPI_Barrier(mpi_communicator);

    for (unsigned int r = 0; r < n_proc; ++r)
    {
        MPI_Barrier(mpi_communicator);

        if (my_rank == r)
        {
            std::cout << "Rank " << my_rank
                      << ": locally owned active cells = "
                      << triangulation.n_locally_owned_active_cells()
                      << ", global active cells = "
                      << triangulation.n_global_active_cells()
                      << std::endl;
        }
        MPI_Barrier(mpi_communicator);
    }

    if (topt.write_loaded_tria)
    {
        GridOut grid_out;

        grid_out.write_mesh_per_processor_as_vtu(triangulation,
                                                 dbg_base);

        pcout << "Wrote debug VTU mesh: "
              << dbg_base
              << std::endl;
    }
}


template <int dim>
void NPSAT_TRACE<dim>::read_parallel_coarse_tria_from_files() {
    const unsigned int vpc =GeometryInfo<dim>::vertices_per_cell;

    unsigned int n_vertices = 0;
    unsigned int n_cells    = 0;

    std::vector<double> vertex_coords;
    std::vector<unsigned int> cell_vertices;

    int ok = 1;
    if (my_rank == 0) {
        const std::string vertex_file = topt.input_prefix + "_coarse_tria_vertices.dat";
        const std::string cell_file   = topt.input_prefix + "_coarse_tria_cells.dat";

        { // Read the vertices
            std::ifstream in(vertex_file.c_str());
            if (!in.good())
                ok = 0;
            unsigned int file_dim = 0;

            if (ok){
                in >> n_vertices >> file_dim;
                if (file_dim != dim)
                    ok = 0;
            }
            if (ok) {
                vertex_coords.resize(n_vertices * dim, 0.0);
                for (unsigned int i = 0; i < n_vertices; ++i) {
                    unsigned int id;
                    in >> id;
                    if (id >= n_vertices) {
                        ok = 0;
                        break;
                    }
                    for (unsigned int d = 0; d < dim; ++d)
                        in >> vertex_coords[id * dim + d];
                }
            }
        }

        {// Read the cells
            std::ifstream in(cell_file.c_str());
            if (!in.good())
                ok = 0;

            unsigned int file_vpc = 0;
            if (ok) {
                in >> n_cells >> file_vpc;
                if (file_vpc != vpc)
                    ok = 0;
            }
            if (ok) {
                cell_vertices.resize(n_cells * vpc, 0);
                for (unsigned int c = 0; c < n_cells; ++c) {
                    unsigned int id;
                    in >> id;
                    if (id >= n_cells) {
                        ok = 0;
                        break;
                    }
                    for (unsigned int j = 0; j < vpc; ++j)
                        in >> cell_vertices[id * vpc + j];
                }
            }
        }
    }


    MPI_Bcast(&ok, 1, MPI_INT, 0, mpi_communicator);
    AssertThrow(ok == 1,ExcMessage("Error reading coarse triangulation files with prefix: " + topt.input_prefix));


    MPI_Bcast(&n_vertices, 1, MPI_UNSIGNED, 0, mpi_communicator);
    MPI_Bcast(&n_cells,    1, MPI_UNSIGNED, 0, mpi_communicator);

    if (my_rank != 0)
    {
        vertex_coords.resize(n_vertices * dim);
        cell_vertices.resize(n_cells * vpc);
    }

    MPI_Bcast(&vertex_coords[0],
              static_cast<int>(vertex_coords.size()),
              MPI_DOUBLE,
              0,
              mpi_communicator);

    MPI_Bcast(&cell_vertices[0],
              static_cast<int>(cell_vertices.size()),
              MPI_UNSIGNED,
              0,
              mpi_communicator);

    std::vector<Point<dim> > vertices(n_vertices);


    for (unsigned int i = 0; i < n_vertices; ++i)
        for (unsigned int d = 0; d < dim; ++d)
            vertices[i][d] = vertex_coords[i * dim + d];

    std::vector<CellData<dim> > cells(n_cells);

    for (unsigned int c = 0; c < n_cells; ++c){
        for (unsigned int j = 0; j < vpc; ++j)
            cells[c].vertices[j] = cell_vertices[c * vpc + j];
        cells[c].material_id = 0;
    }

    SubCellData subcelldata;
    triangulation.clear();
    triangulation.create_triangulation(vertices, cells, subcelldata);
    MPI_Barrier(mpi_communicator);

    pcout << "Rebuilt distributed coarse triangulation from prefix: " << topt.input_prefix << std::endl;
}

template <int dim>
void NPSAT_TRACE<dim>::setup_triangulation_helpers() {

    // Local description of the locally owned region as bounding boxes.
    // Deal.II expects "a vector of BoundingBox", because a rank's owned region
    // can be disconnected => multiple boxes.
    const std::vector<BoundingBox<dim>> local_bboxes =
        GridTools::compute_mesh_predicate_bounding_box(
            triangulation,IteratorFilters::LocallyOwnedCell());

    // Gather to rank-vector: global_bounding_boxes[rank] = that rank's vector of bboxes.
    global_bounding_boxes = Utilities::MPI::all_gather(mpi_communicator, local_bboxes);

    AssertThrow(global_bounding_boxes.size() == n_proc, ExcMessage("global_bounding_boxes size mismatch with n_proc."));

    pcout << "Built global_bounding_boxes for particle insertion: "
          << "rank0 has " << global_bounding_boxes[0].size() << " boxes, "
          << "this rank has " << local_bboxes.size() << " boxes."
          << std::endl;
}

template<int dim>
void NPSAT_TRACE<dim>::setup_system() {

    dof_handler_flux.distribute_dofs(fe_flux);
    locally_owned_dofs = dof_handler_flux.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler_flux,locally_relevant_dofs);

    // Initialize RT0 face velocity vector (normal velocity / flux density per RT0 face dof)
    vface.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

    // Assign user_index(slot) deterministically on locally owned active cells
    triangulation.clear_user_data();
    unsigned int slot = 0;
    for (auto cell = triangulation.begin_active(); cell != triangulation.end(); ++cell)
        if (cell->is_locally_owned())
            cell->set_user_index(slot++);
    AssertThrow(slot == triangulation.n_locally_owned_active_cells(), ExcInternalError());

    // Load static mapping:
    // - owned_gids_ref
    // - rt0_face_gid
    // - rt0_face_sign
    // - rt0_face_owned_pos
    // - rt0_face_flags
    // - cellid_to_slot
    //
    // rt0_face_flags are used by the tracer to distinguish:
    //   bit 0 (0x1): coarse-side parent face at a coarse-fine interface
    //   bit 1 (0x2): refined-side active child face at a coarse-fine interface
    load_velocity_io_mapping();

    // Build slot -> CellId lookup once.
    const auto n_local_cells = triangulation.n_locally_owned_active_cells();
    slot_cellid.resize(n_local_cells);

    for (auto cell = triangulation.begin_active(); cell != triangulation.end(); ++cell)
    {
        if (!cell->is_locally_owned())
            continue;
        const unsigned int s = static_cast<unsigned int>(cell->user_index());
        slot_cellid[s] = cell->id();
    }

    // Lazy per-cell cache. Rebuilt when needed after each new velocity step is loaded.
    all_cells_cache.resize(n_local_cells);
    all_cells_cache_valid.assign(n_local_cells, false);

    slot_water_table_elevation.assign(n_local_cells, std::numeric_limits<double>::quiet_NaN());

    read_cell_well_map_binary_once();
}

template<int dim>
void NPSAT_TRACE<dim>::load_velocity_io_mapping() {
    MPI_Barrier(mpi_communicator);
    const unsigned int faces_per_cell = GeometryInfo<dim>::faces_per_cell;

    const std::string str_rank = Utilities::int_to_string(my_rank, 4);
    const std::string filename = topt.input_prefix + "_velmap_rank_" + str_rank + ".bin";

    pcout << "Loading velocity IO mapping: " << filename << std::endl;
    std::ifstream in(filename, std::ios::binary);
    AssertThrow(in.good(), ExcMessage("Could not open mapping file: " + filename));

    std::uint64_t magic = 0;
    std::uint32_t version = 0;
    std::uint32_t dim_u32 = 0;
    std::uint32_t fpc_u32 = 0;
    std::uint64_t n_global_dofs_file = 0;
    std::uint64_t n_owned_dofs_file  = 0;
    std::uint64_t n_local_cells_file = 0;

    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&dim_u32), sizeof(dim_u32));
    in.read(reinterpret_cast<char*>(&fpc_u32), sizeof(fpc_u32));
    in.read(reinterpret_cast<char*>(&n_global_dofs_file), sizeof(n_global_dofs_file));
    in.read(reinterpret_cast<char*>(&n_owned_dofs_file), sizeof(n_owned_dofs_file));
    in.read(reinterpret_cast<char*>(&n_local_cells_file), sizeof(n_local_cells_file));

    const std::uint64_t expected_magic = 0x4E50534154564D50ull; // must match writer
    AssertThrow(magic == expected_magic, ExcMessage("Bad magic in mapping file: " + filename));
    AssertThrow(version == 2, ExcMessage("Unsupported mapping file version: " + filename));
    AssertThrow(dim_u32 == dim, ExcMessage("Mapping dim mismatch: " + filename));
    AssertThrow(fpc_u32 == faces_per_cell, ExcMessage("faces_per_cell mismatch: " + filename));
    AssertThrow(n_global_dofs_file == dof_handler_flux.n_dofs(), ExcMessage("Global RT0 DoF count mismatch; incompatible mesh/DoFs."));
    AssertThrow(n_owned_dofs_file == locally_owned_dofs.n_elements(), ExcMessage("Owned DoF count mismatch; incompatible partitioning/DoFs."));

    const auto n_local_cells_now = triangulation.n_locally_owned_active_cells();
    AssertThrow(n_local_cells_file == static_cast<std::uint64_t>(n_local_cells_now), ExcMessage("Local owned cell count mismatch; incompatible partitioning."));

    // --- Ensure user_index(slot) exists on trace side too ---
    // Ensure deterministic slot numbering on the trace side.
    triangulation.clear_user_data();
    unsigned int slot = 0;
    for (auto cell = triangulation.begin_active(); cell != triangulation.end(); ++cell)
        if (cell->is_locally_owned())
            cell->set_user_index(slot++);
    AssertThrow(slot == n_local_cells_now, ExcInternalError());

    // Build CellId -> slot map.
    cellid_to_slot.clear();
    cellid_to_slot.reserve(static_cast<std::size_t>(n_local_cells_now) * 2);
    for (auto cell = triangulation.begin_active(); cell != triangulation.end(); ++cell)
        if (cell->is_locally_owned())
            cellid_to_slot.emplace(cell->id().to_string(), static_cast<unsigned int>(cell->user_index()));

    // --- Read owned gids in file order ---
    rt0_map.clear();
    rt0_map.owned_gids().assign(static_cast<std::size_t>(n_owned_dofs_file), 0);
    for (std::size_t i = 0; i < rt0_map.owned_gids().size(); ++i){
        std::uint64_t gid64 = 0;
        in.read(reinterpret_cast<char*>(&gid64), sizeof(gid64));
        rt0_map.owned_gids()[i] = static_cast<types::global_dof_index>(gid64);
    }
    AssertThrow(in.good(), ExcMessage("Truncated mapping file (owned gids): " + filename));

    // Strong check: order must match IndexSet begin/end now (this is what the values-only IO assumes)
    {
        auto it = locally_owned_dofs.begin();
        for (std::size_t i = 0; i < rt0_map.owned_gids().size(); ++i, ++it)
            AssertThrow(static_cast<dealii::types::global_dof_index>(*it) == rt0_map.owned_gids()[i],
                        ExcMessage("Owned DoF ordering mismatch; values-only files cannot be read safely."));
    }

    // Build gid -> local owned position lookup.
    std::unordered_map<types::global_dof_index, std::uint32_t> gid_to_owned_pos;
    gid_to_owned_pos.reserve(rt0_map.owned_gids().size() * 2);
    for (std::uint32_t i = 0; i < rt0_map.owned_gids().size(); ++i)
        gid_to_owned_pos.emplace(rt0_map.owned_gids()[i], i);

    // Allocate per-cell face tables.
    rt0_map.resize(n_local_cells_now);

    // --- Read cell records and place by slot ---
    for (std::uint64_t k = 0; k < n_local_cells_file; ++k)
    {
        std::uint16_t len = 0;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        AssertThrow(in.good(), ExcMessage("Truncated mapping file (cid len): " + filename));

        std::string cid;
        cid.resize(len);
        if (len > 0)
            in.read(&cid[0], static_cast<std::streamsize>(len));

        AssertThrow(in.good(), ExcMessage("Truncated mapping file (cid): " + filename));

        auto it_slot = cellid_to_slot.find(cid);
        AssertThrow(it_slot != cellid_to_slot.end(), ExcMessage("Mapping references a non-local cell on this rank: " + cid));

        const unsigned int s = it_slot->second;

        for (unsigned int f = 0; f < faces_per_cell; ++f)
        {
            std::uint64_t gid64 = 0;
            std::int8_t   sgn8  = 0;
            std::uint8_t  flg8  = 0;

            in.read(reinterpret_cast<char*>(&gid64), sizeof(gid64));
            in.read(reinterpret_cast<char*>(&sgn8),  sizeof(sgn8));
            in.read(reinterpret_cast<char*>(&flg8),  sizeof(flg8));
            AssertThrow(in.good(), ExcMessage("Truncated mapping file (face rec): " + filename));

            const auto gid = static_cast<dealii::types::global_dof_index>(gid64);

            AssertThrow(sgn8 == 1 || sgn8 == -1, ExcMessage("Invalid RT0 face sign in mapping file."));

            // Allowed flag combinations:
            //   0x0 : regular face / boundary
            //   0x1 : coarse-side parent face
            //   0x2 : refined-side child face
            AssertThrow((flg8 & ~std::uint8_t(0x3)) == 0, ExcMessage("Invalid RT0 face flags in mapping file."));

            rt0_map.gids(s)[f]  = gid;
            rt0_map.signs(s)[f] = sgn8;
            rt0_map.flags(s)[f] = flg8;

            auto itp = gid_to_owned_pos.find(gid);
            rt0_map.owned_positions(s)[f] = (itp == gid_to_owned_pos.end())
                                         ? -1
                                         : static_cast<std::int32_t>(itp->second);
        }
    }
    AssertThrow(in.good(), ExcMessage("Error while reading mapping file: " + filename));
    MPI_Barrier(mpi_communicator);
    pcout << "\t Done loading velocity IO mapping: " << filename << std::endl;
    MPI_Barrier(mpi_communicator);
}

template <int dim>
void NPSAT_TRACE<dim>::read_cell_well_map_binary_once() {
    MPI_Barrier(mpi_communicator);
    const std::string rank_s = Utilities::int_to_string(my_rank, 4);
    const std::string fname  = topt.input_prefix + "_cellwell_map_rank_" + rank_s + ".bin";

    pcout << "Loading cell - well mapping: " << fname << std::endl;

    std::ifstream in(fname, std::ios::binary);
    if (!in.good())
        throw std::runtime_error("Could not open: " + fname);

    // header magic[8] = {'C','W','M','A','P','v','2','\0'}
    char magic[8];
    in.read(magic, 8);
    if (!in.good())
        throw std::runtime_error("Failed reading header: " + fname);

    const char expect_magic[8] = {'C','W','M','A','P','v','2','\0'};
    if (std::memcmp(magic, expect_magic, 8) != 0)
        throw std::runtime_error("Bad magic in: " + fname);

    std::uint32_t version = 0;
    std::uint32_t dim_u32 = 0;
    npsat_trace::read_pod(in, version);
    npsat_trace::read_pod(in, dim_u32);

    if (version != 2)
        throw std::runtime_error("Unsupported cell-well map version in: " + fname);
    if (dim_u32 != static_cast<std::uint32_t>(dim))
        throw std::runtime_error("Dimension mismatch in: " + fname);

    std::uint64_t n_cells = 0;
    npsat_trace::read_pod(in, n_cells);

    // Prepare storage aligned with the tracer’s cell-slot layout
    slot_cell_well_links.clear();
    slot_cell_well_links.resize(slot_cellid.size());

    for (std::uint64_t ic = 0; ic < n_cells; ++ic) {

        const std::string cell_id_str = npsat_trace::read_string(in);

        std::uint32_t n_links = 0;
        npsat_trace::read_pod(in, n_links);

        // Map the cell_id_str to tracer slot (if present on this rank)
        auto it_slot = cellid_to_slot.find(cell_id_str);
        const bool have_slot = (it_slot != cellid_to_slot.end());
        const unsigned int slot = have_slot ? it_slot->second : 0;

        std::vector<npsat_trace::CellWellLink> links;
        links.reserve(n_links);

        for (std::uint32_t j = 0; j < n_links; ++j) {
            npsat_trace::CellWellLink L;

            std::uint32_t wgid = 0;
            npsat_trace::read_pod(in, wgid);
            L.well_global_index = wgid;

            std::uint16_t owner = 0;
            npsat_trace::read_pod(in, owner);
            L.well_owner_rank = owner;

            npsat_trace::read_pod(in, L.eid);
            npsat_trace::read_pod(in, L.wx);
            npsat_trace::read_pod(in, L.wy);
            npsat_trace::read_pod(in, L.wtop);
            npsat_trace::read_pod(in, L.wbot);
            npsat_trace::read_pod(in, L.q_row);

            npsat_trace::read_pod(in, L.ze);
            npsat_trace::read_pod(in, L.sl);
            npsat_trace::read_pod(in, L.w_zbot);

            links.push_back(L);
        }

        // Store only cells that exist in this tracer rank’s slot mapping.
        // This keeps the tracer robust even if the file contains entries you do not use.
        if (have_slot)
            slot_cell_well_links[slot] = std::move(links);
    }
    if (!in.good())
        throw std::runtime_error("Read failed (cell-well map): " + fname);

    MPI_Barrier(mpi_communicator);
    pcout << "\t Done loading cell - well mapping: " << fname << std::endl;
    MPI_Barrier(mpi_communicator);
}


#endif //NPSAT_TRACE_LOAD_IMPL_H
