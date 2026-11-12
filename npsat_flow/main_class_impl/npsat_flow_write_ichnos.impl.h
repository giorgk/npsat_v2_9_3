//
// Steady-state velocity-cloud and cell-graph export for redist/Ichnos.
//

#ifndef NPSAT_FLOW_WRITE_ICHNOS_IMPL_H
#define NPSAT_FLOW_WRITE_ICHNOS_IMPL_H

template <int dim>
void NPSAT_FLOW<dim>::save_ichnos_data(const std::string &prefix) const
{
    if (!uo.save_ichnos_data)
        return;

    AssertThrow(uo.sim_opt.steady_state,
                ExcMessage("The redist/Ichnos export is available only for steady-state simulations."));
    AssertThrow(degree == 0,
                ExcMessage("The redist/Ichnos exporter currently requires RT0 fluxes."));
    AssertThrow(uo.ichnos_points_per_cell == 1 ||
                    uo.ichnos_points_per_cell == 8,
                ExcMessage("Ichnos points per cell must be 1 or 8."));

    const bool binary_output = (uo.ichnos_format == "binary");
    const bool coarse_mode = (uo.ichnos_face_mode == "coarse");
    const unsigned int faces_per_cell = GeometryInfo<dim>::faces_per_cell;
    const unsigned int vertices_per_cell = GeometryInfo<dim>::vertices_per_cell;
    const unsigned int n_flux_dofs = fe_flux.n_dofs_per_cell();

    AssertThrow(n_flux_dofs == faces_per_cell,
                ExcMessage("The redist/Ichnos exporter expects one RT0 DoF per face."));

    typedef typename DoFHandler<dim>::active_cell_iterator FluxCell;
    typedef typename DoFHandler<dim>::cell_iterator FluxCellIterator;

    struct GraphRecord
    {
        std::string cell_id;
        Point<dim> center;
        std::vector<std::string> neighbors;
    };

    struct VelocityRecord
    {
        std::string cell_id;
        Point<dim> point;
        Tensor<1, dim> velocity;
        double diameter;
        double ratio;
    };

    std::vector<FluxCell> owned_cells;
    owned_cells.reserve(triangulation.n_locally_owned_active_cells());
    for (const auto &cell : dof_handler_flux.active_cell_iterators())
        if (cell->is_locally_owned())
            owned_cells.push_back(cell);

    std::sort(owned_cells.begin(), owned_cells.end(),
              [](const FluxCell &a, const FluxCell &b)
              {
                  return a->id().to_string() < b->id().to_string();
              });

    // Build the legacy redist stencil: cells sharing a vertex are graph
    // neighbors. Locally owned and ghost cells supply cross-rank CellIds.
    std::unordered_map<unsigned int, std::vector<std::string>> cells_at_vertex;
    for (const auto &cell : dof_handler_flux.active_cell_iterators())
    {
        if (cell->is_artificial())
            continue;
        const std::string id = cell->id().to_string();
        for (unsigned int v = 0; v < vertices_per_cell; ++v)
            cells_at_vertex[cell->vertex_index(v)].push_back(id);
    }

    std::vector<GraphRecord> graph_records;
    graph_records.reserve(owned_cells.size());
    for (const auto &cell : owned_cells)
    {
        GraphRecord rec;
        rec.cell_id = cell->id().to_string();
        rec.center = cell->center();

        std::set<std::string> neighbor_ids;
        for (unsigned int v = 0; v < vertices_per_cell; ++v)
        {
            const auto found = cells_at_vertex.find(cell->vertex_index(v));
            if (found == cells_at_vertex.end())
                continue;
            for (const auto &neighbor_id : found->second)
                if (neighbor_id != rec.cell_id)
                    neighbor_ids.insert(neighbor_id);
        }
        rec.neighbors.assign(neighbor_ids.begin(), neighbor_ids.end());
        graph_records.push_back(rec);
    }

    QGauss<dim-1> face_quadrature(fe_flux.degree + 2);
    FEFaceValues<dim> fe_face(fe_flux, face_quadrature,
                              update_values | update_normal_vectors |
                                  update_JxW_values);
    FEValuesExtractors::Vector flux(0);

    const auto outward_face_average =
        [&](const FluxCellIterator &cell, const unsigned int face_no) -> double
        {
            std::vector<types::global_dof_index> dof_indices(n_flux_dofs);
            cell->get_dof_indices(dof_indices);
            Vector<double> coefficients(n_flux_dofs);
            for (unsigned int i = 0; i < n_flux_dofs; ++i)
            {
                AssertThrow(flux_locally_relevant_dofs.is_element(dof_indices[i]),
                            ExcMessage("A neighboring RT0 DoF required by the Ichnos exporter is not locally relevant."));
                coefficients(i) = q_new[dof_indices[i]];
            }

            fe_face.reinit(cell, face_no);
            double flux_integral = 0.0;
            double area = 0.0;
            for (unsigned int q = 0; q < face_quadrature.size(); ++q)
            {
                Tensor<1, dim> qh;
                for (unsigned int i = 0; i < n_flux_dofs; ++i)
                    qh += coefficients(i) * fe_face[flux].value(i, q);
                flux_integral +=
                    (qh * fe_face.normal_vector(q)) * fe_face.JxW(q);
                area += fe_face.JxW(q);
            }
            AssertThrow(area > 0.0, ExcMessage("Zero-area face in the Ichnos exporter."));
            return flux_integral / area;
        };

    QGauss<dim> point_quadrature(uo.ichnos_points_per_cell == 1 ? 1 : 2);
    FEValues<dim> fe_points(fe_flux, point_quadrature,
                            update_values | update_quadrature_points);

    std::vector<VelocityRecord> velocity_records;
    velocity_records.reserve(owned_cells.size() *
                             static_cast<std::size_t>(uo.ichnos_points_per_cell));

    for (const auto &cell : owned_cells)
    {
        std::vector<types::global_dof_index> dof_indices(n_flux_dofs);
        cell->get_dof_indices(dof_indices);

        Vector<double> native_coefficients(n_flux_dofs);
        for (unsigned int i = 0; i < n_flux_dofs; ++i)
        {
            AssertThrow(flux_locally_relevant_dofs.is_element(dof_indices[i]),
                        ExcMessage("A local RT0 DoF required by the Ichnos exporter is not locally relevant."));
            native_coefficients(i) = q_new[dof_indices[i]];
        }

        // M maps local RT0 coefficients to outward face-average velocities.
        FullMatrix<double> face_matrix(faces_per_cell, n_flux_dofs);
        for (unsigned int f = 0; f < faces_per_cell; ++f)
        {
            fe_face.reinit(cell, f);
            double area = 0.0;
            for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                area += fe_face.JxW(q);
            AssertThrow(area > 0.0, ExcMessage("Zero-area face in the Ichnos exporter."));

            for (unsigned int i = 0; i < n_flux_dofs; ++i)
                for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                    face_matrix(f, i) +=
                        (fe_face[flux].value(i, q) * fe_face.normal_vector(q)) *
                        fe_face.JxW(q) / area;
        }

        Vector<double> selected_face_velocity(faces_per_cell);
        face_matrix.vmult(selected_face_velocity, native_coefficients);

        for (unsigned int f = 0; f < faces_per_cell; ++f)
        {
            if (cell->at_boundary(f))
                continue;

            if (coarse_mode && cell->neighbor_is_coarser(f))
            {
                const FluxCellIterator coarse_cell = cell->neighbor(f);
                AssertThrow(coarse_cell->is_active() &&
                                !coarse_cell->is_artificial(),
                            ExcMessage("The coarse neighbor required by the Ichnos exporter is unavailable."));
                const unsigned int coarse_face = cell->neighbor_face_no(f);
                selected_face_velocity(f) =
                    -outward_face_average(coarse_cell, coarse_face);
            }
            else if (!coarse_mode && !cell->neighbor(f)->is_active())
            {
                // This is the coarse side of a hanging interface. Aggregate
                // authoritative fine-child fluxes back to the parent face.
                const unsigned int n_subfaces = cell->face(f)->n_children();
                double parent_flux = 0.0;
                double parent_area = 0.0;

                for (unsigned int subface = 0; subface < n_subfaces; ++subface)
                {
                    const FluxCellIterator child_cell =
                        cell->neighbor_child_on_subface(f, subface);
                    AssertThrow(child_cell->is_active() &&
                                    !child_cell->is_artificial(),
                                ExcMessage("A fine neighbor required by the Ichnos exporter is unavailable."));

                    unsigned int child_face = numbers::invalid_unsigned_int;
                    for (unsigned int cf = 0; cf < faces_per_cell; ++cf)
                        if (!child_cell->at_boundary(cf) &&
                            child_cell->neighbor_is_coarser(cf) &&
                            child_cell->neighbor(cf)->id() == cell->id())
                        {
                            child_face = cf;
                            break;
                        }
                    AssertThrow(child_face != numbers::invalid_unsigned_int,
                                ExcMessage("Could not match a fine face to its coarse neighbor."));

                    const double child_area = child_cell->face(child_face)->measure();
                    parent_flux +=
                        -outward_face_average(child_cell, child_face) * child_area;
                    parent_area += child_area;
                }

                AssertThrow(parent_area > 0.0,
                            ExcMessage("Zero-area coarse-fine interface in the Ichnos exporter."));
                selected_face_velocity(f) = parent_flux / parent_area;
            }
        }

        FullMatrix<double> inverse_face_matrix(n_flux_dofs, n_flux_dofs);
        inverse_face_matrix.invert(face_matrix);
        Vector<double> selected_coefficients(n_flux_dofs);
        inverse_face_matrix.vmult(selected_coefficients, selected_face_velocity);

        const std::pair<Point<dim>, Point<dim>> bounds =
            cell->bounding_box().get_boundary_points();
        const double dx = bounds.second[0] - bounds.first[0];
        const double dy = bounds.second[1] - bounds.first[1];
        const double dz = bounds.second[dim-1] - bounds.first[dim-1];
        const double diameter = std::max(dx, dy);
        AssertThrow(diameter > 0.0 && dz > 0.0,
                    ExcMessage("Invalid cell dimensions in the Ichnos exporter."));
        const double ratio = diameter / dz;

        fe_points.reinit(cell);
        const std::vector<Point<dim>> &points = fe_points.get_quadrature_points();
        for (unsigned int q = 0; q < point_quadrature.size(); ++q)
        {
            VelocityRecord rec;
            rec.cell_id = cell->id().to_string();
            rec.point = points[q];
            rec.diameter = diameter;
            rec.ratio = ratio;
            for (unsigned int i = 0; i < n_flux_dofs; ++i)
                rec.velocity += selected_coefficients(i) *
                                fe_points[flux].value(i, q);
            velocity_records.push_back(rec);
        }
    }

    const std::string rank_string = Utilities::int_to_string(my_rank, 4);
    const std::string file_base = prefix + "_4redist_" + rank_string;
    const unsigned int batch_size = uo.velocity_write_batch_size;
    const unsigned int n_batches = (n_proc + batch_size - 1) / batch_size;

    pcout << "Writing steady-state redist/Ichnos "
          << uo.ichnos_format << " files with "
          << uo.ichnos_points_per_cell << " point(s) per cell and "
          << uo.ichnos_face_mode << "-face velocities." << std::endl;

    for (unsigned int batch = 0; batch < n_batches; ++batch)
    {
        const unsigned int first_rank = batch * batch_size;
        const unsigned int last_rank = std::min(n_proc, first_rank + batch_size);

        if (my_rank >= first_rank && my_rank < last_rank)
        {
            if (!binary_output)
            {
                const std::string velocity_name = file_base + ".vel";
                std::ofstream velocity_out(velocity_name.c_str());
                AssertThrow(velocity_out.good(),
                            ExcMessage("Could not open Ichnos velocity file: " + velocity_name));
                velocity_out << std::scientific << std::setprecision(16);
                for (const auto &rec : velocity_records)
                {
                    velocity_out << rec.point[0] << ' '
                                 << rec.point[1] << ' '
                                 << rec.point[2] << ' '
                                 << rec.velocity[0] << ' '
                                 << rec.velocity[1] << ' '
                                 << rec.velocity[2] << ' '
                                 << my_rank << ' '
                                 << rec.diameter << ' '
                                 << rec.ratio << ' '
                                 << rec.cell_id << '\n';
                }
                AssertThrow(velocity_out.good(),
                            ExcMessage("Failed writing Ichnos velocity file: " + velocity_name));

                const std::string graph_name = file_base + ".grph";
                std::ofstream graph_out(graph_name.c_str());
                AssertThrow(graph_out.good(),
                            ExcMessage("Could not open Ichnos graph file: " + graph_name));
                graph_out << std::scientific << std::setprecision(16);
                for (const auto &rec : graph_records)
                {
                    graph_out << rec.center[0] << ' '
                              << rec.center[1] << ' '
                              << rec.center[2] << ' '
                              << rec.neighbors.size() << ' '
                              << rec.cell_id;
                    for (const auto &neighbor_id : rec.neighbors)
                        graph_out << ' ' << neighbor_id;
                    graph_out << '\n';
                }
                AssertThrow(graph_out.good(),
                            ExcMessage("Failed writing Ichnos graph file: " + graph_name));
            }
            else
            {
                const std::string velocity_name = file_base + ".vel.bin";
                std::ofstream velocity_out(velocity_name.c_str(), std::ios::binary);
                AssertThrow(velocity_out.good(),
                            ExcMessage("Could not open binary Ichnos velocity file: " + velocity_name));
                const char velocity_magic[8] = {'N','P','S','I','V','L','1','\0'};
                velocity_out.write(velocity_magic, 8);
                const std::uint32_t version = 1;
                const std::uint32_t dim_u32 = dim;
                const std::uint32_t rank_u32 = my_rank;
                const std::uint32_t nproc_u32 = n_proc;
                const std::uint32_t points_u32 = uo.ichnos_points_per_cell;
                const std::uint32_t face_mode_u32 = coarse_mode ? 1u : 0u;
                const std::uint64_t n_cells =
                    static_cast<std::uint64_t>(graph_records.size());
                const std::uint64_t n_points =
                    static_cast<std::uint64_t>(velocity_records.size());
                npsat_flow::write_pod(velocity_out, version);
                npsat_flow::write_pod(velocity_out, dim_u32);
                npsat_flow::write_pod(velocity_out, rank_u32);
                npsat_flow::write_pod(velocity_out, nproc_u32);
                npsat_flow::write_pod(velocity_out, points_u32);
                npsat_flow::write_pod(velocity_out, face_mode_u32);
                npsat_flow::write_pod(velocity_out, n_cells);
                npsat_flow::write_pod(velocity_out, n_points);
                for (const auto &rec : velocity_records)
                {
                    for (unsigned int d = 0; d < dim; ++d)
                        npsat_flow::write_pod(velocity_out, rec.point[d]);
                    for (unsigned int d = 0; d < dim; ++d)
                        npsat_flow::write_pod(velocity_out, rec.velocity[d]);
                    const std::uint32_t source_rank = my_rank;
                    npsat_flow::write_pod(velocity_out, source_rank);
                    npsat_flow::write_pod(velocity_out, rec.diameter);
                    npsat_flow::write_pod(velocity_out, rec.ratio);
                    npsat_flow::write_string(velocity_out, rec.cell_id);
                }
                AssertThrow(velocity_out.good(),
                            ExcMessage("Failed writing binary Ichnos velocity file: " + velocity_name));

                const std::string graph_name = file_base + ".grph.bin";
                std::ofstream graph_out(graph_name.c_str(), std::ios::binary);
                AssertThrow(graph_out.good(),
                            ExcMessage("Could not open binary Ichnos graph file: " + graph_name));
                const char graph_magic[8] = {'N','P','S','I','G','R','1','\0'};
                graph_out.write(graph_magic, 8);
                npsat_flow::write_pod(graph_out, version);
                npsat_flow::write_pod(graph_out, dim_u32);
                npsat_flow::write_pod(graph_out, rank_u32);
                npsat_flow::write_pod(graph_out, nproc_u32);
                npsat_flow::write_pod(graph_out, n_cells);
                for (const auto &rec : graph_records)
                {
                    for (unsigned int d = 0; d < dim; ++d)
                        npsat_flow::write_pod(graph_out, rec.center[d]);
                    const std::uint32_t n_neighbors =
                        static_cast<std::uint32_t>(rec.neighbors.size());
                    npsat_flow::write_pod(graph_out, n_neighbors);
                    npsat_flow::write_string(graph_out, rec.cell_id);
                    for (const auto &neighbor_id : rec.neighbors)
                        npsat_flow::write_string(graph_out, neighbor_id);
                }
                AssertThrow(graph_out.good(),
                            ExcMessage("Failed writing binary Ichnos graph file: " + graph_name));
            }
        }

        MPI_Barrier(mpi_communicator);
    }
}

#endif // NPSAT_FLOW_WRITE_ICHNOS_IMPL_H
