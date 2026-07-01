//
// Created by giorgk on 6/24/26.
//

#ifndef MESH_GEN_H
#define MESH_GEN_H

#include <deal.II/grid/grid_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>

#include "flow_structures.h"
#include "mnwells.h"
#include "streams.h"
#include "BC/dirichlet_bc.h"
#include "BC/ghb_bc.h"
#include "boost_helper_func.h"

namespace npsat_flow {
    using namespace dealii;

    template <int dim>
    class GridBuilder{
    public:
        static void build(
            parallel::distributed::Triangulation<dim>& triangulation,
            const user_options& uo,
            const MPI_Comm& mpi_communicator)
        {
            if (uo.isBox)
                build_box_grid(triangulation, uo, mpi_communicator);
            else
                build_file_grid(triangulation, uo, mpi_communicator);
        }

        static constexpr int spacedim = dim;
        static constexpr int basedim = dim - 1;

        using TriaType = parallel::distributed::Triangulation<spacedim>;
        using BaseTriaType = Triangulation<basedim>;
        using CellIterator = typename TriaType::active_cell_iterator;

        struct RefinementTargets
        {
            MNWellCollection *mnwells = nullptr;
            StreamCollection<dim> *streams = nullptr;

            const DirichletBoundary<dim> *dirichlet = nullptr;
            const GHBBoundary<dim> *ghb = nullptr;
            //FacePredicate neumann_face;
        };

        static void refine_triangulation(
            TriaType &triangulation,
            const user_options &uo,
            const RefinementTargets &targets,
            const MPI_Comm &mpi_communicator);

        static void write_parallel_coarse_tria_to_files(const TriaType &tria, const std::string &prefix, const MPI_Comm &mpi_communicator);

    private:
        static void build_box_grid(TriaType &triangulation, const user_options& uo,const MPI_Comm& mpi_communicator);
        static void build_file_grid(TriaType &triangulation, const user_options& uo, const MPI_Comm& mpi_communicator);
        static void read_2d_grid(BaseTriaType &triangulation, const user_options& uo);
        static double signed_polygon_area(const std::vector<Point<basedim>> &vertices,
                                          const unsigned int cell_vertices[4]);
        static void convert_to_parallel(const Triangulation<dim>& tria3D, TriaType& triangulation);
        static void assign_default_boundary_ids(TriaType& triangulation);

        struct SurfaceEvaluator
        {
            bool enabled = false;
            double constant = 0.0;
            std::shared_ptr<InterpInterface<dim>> interp;

            double value(const Point<dim> &p, const double fallback) const
            {
                if (!enabled)
                    return fallback;

                if (interp)
                    return interp->interpolate(p, 0);

                return constant;
            }
        };

        static SurfaceEvaluator make_surface_evaluator(const std::string &spec,
                                                       const MPI_Comm &mpi_communicator,
                                                       const std::string &input_path);

        static std::shared_ptr<InterpInterface<dim>> make_vertical_distribution(
            const user_options &uo,
            const MPI_Comm &mpi_communicator,
            const std::string &input_path);

        static unsigned int vertical_level_count(
            const user_options &uo,
            const std::shared_ptr<InterpInterface<dim>> &vertical_distribution,
            const unsigned int fallback_level_count);

        static void conform_to_top_bottom(
            TriaType &triangulation,
            const user_options &uo,
            const SurfaceEvaluator &top,
            const SurfaceEvaluator &bottom,
            const std::shared_ptr<InterpInterface<dim>> &vertical_distribution,
            const double source_bottom,
            const double source_top,
            const unsigned int n_vertical_levels);

        static void mark_top_neighborhood_from_cell(const CellIterator& cell, const unsigned int depth);
        static bool cell_has_top_boundary(const CellIterator& cell);
        static bool cell_has_other_boundary(const CellIterator& cell);
        static bool cell_contains_well(const CellIterator& cell, MNWellCollection& mnwells);
        static bool cell_has_top_stream(const CellIterator& cell, StreamCollection<dim>& streams);
        static bool cell_has_dirichlet_boundary(const CellIterator &cell,
                                                const DirichletBoundary<dim> &dirichlet_bc);
        static bool cell_has_ghb_boundary(const CellIterator &cell,
                                          const GHBBoundary<dim> &ghb_bc);
    };

    template <int dim>
    typename GridBuilder<dim>::SurfaceEvaluator
    GridBuilder<dim>::make_surface_evaluator(const std::string &spec,
                                             const MPI_Comm &mpi_communicator,
                                             const std::string &input_path)
    {
        SurfaceEvaluator evaluator;
        const std::string clean_spec = trim(spec);

        if (clean_spec.empty())
            return evaluator;

        std::string filename;
        double value = 0.0;
        if (parse_double_or_file(clean_spec, value, filename))
        {
            evaluator.enabled = true;
            evaluator.constant = value;
            return evaluator;
        }

        evaluator.enabled = true;
        evaluator.interp = std::make_shared<InterpInterface<dim>>();
        evaluator.interp->read_master_file(filename, 1.0, mpi_communicator, input_path);
        return evaluator;
    }

    template <int dim>
    std::shared_ptr<InterpInterface<dim>>
    GridBuilder<dim>::make_vertical_distribution(const user_options &uo,
                                                 const MPI_Comm &mpi_communicator,
                                                 const std::string &input_path)
    {
        const std::string clean_file = trim(uo.vert_file);
        if (clean_file.empty())
            return nullptr;

        auto interp = std::make_shared<InterpInterface<dim>>();
        interp->read_master_file(clean_file, 1.0, mpi_communicator, input_path);
        return interp;
    }

    template <int dim>
    unsigned int GridBuilder<dim>::vertical_level_count(
        const user_options &uo,
        const std::shared_ptr<InterpInterface<dim>> &vertical_distribution,
        const unsigned int fallback_level_count)
    {
        if (vertical_distribution)
        {
            const std::int64_t n_inner_levels = vertical_distribution->n_times();
            AssertThrow(n_inner_levels >= 0,
                        ExcMessage("Vertical distribution has invalid layer count."));
            return static_cast<unsigned int>(n_inner_levels) + 2;
        }

        if (!uo.vert_discr.empty())
            return static_cast<unsigned int>(uo.vert_discr.size());

        return fallback_level_count;
    }

    template <int dim>
    void GridBuilder<dim>::conform_to_top_bottom(
        TriaType &triangulation,
        const user_options &uo,
        const SurfaceEvaluator &top,
        const SurfaceEvaluator &bottom,
        const std::shared_ptr<InterpInterface<dim>> &vertical_distribution,
        const double source_bottom,
        const double source_top,
        const unsigned int n_vertical_levels)
    {
        const double source_thickness = source_top - source_bottom;
        AssertThrow(std::abs(source_thickness) > 0.0,
                    ExcMessage("Cannot conform mesh with zero source thickness."));

        std::vector<double> conformed_z(triangulation.n_vertices(), 0.0);
        std::vector<bool> vertex_visited(triangulation.n_vertices(), false);

        for (auto cell = triangulation.begin_active(); cell != triangulation.end(); ++cell)
        {
            for (unsigned int vertex_no = 0;
                 vertex_no < GeometryInfo<dim>::vertices_per_cell;
                 ++vertex_no)
            {
                const unsigned int vertex_index = cell->vertex_index(vertex_no);
                if (vertex_visited[vertex_index])
                    continue;

                const Point<dim> &v = cell->vertex(vertex_no);
                Point<dim> p = v;
                p[dim - 1] = 0.0;

                const double z_top = top.value(p, source_top);
                const double z_bot = bottom.value(p, source_bottom);
                const double target_thickness = z_top - z_bot;

                double relative_z = (v[dim - 1] - source_bottom) / source_thickness;
                relative_z = std::max(0.0, std::min(1.0, relative_z));

                if (n_vertical_levels > 1)
                {
                    const unsigned int level =
                        static_cast<unsigned int>(std::lround(relative_z * (n_vertical_levels - 1)));

                    if (vertical_distribution && level > 0 && level + 1 < n_vertical_levels)
                    {
                        // Vertical-distribution files store only intermediate
                        // levels from bottom to top. Bottom and top are omitted
                        // because they are fixed at relative positions 0 and 1.
                        const unsigned int time_index = level - 1;
                        relative_z = vertical_distribution->interpolate(p, time_index);
                    }
                    else if (!vertical_distribution &&
                             uo.vert_discr.size() == n_vertical_levels)
                    {
                        relative_z = uo.vert_discr[level];
                    }

                    relative_z = std::max(0.0, std::min(1.0, relative_z));
                }

                conformed_z[vertex_index] = z_bot + relative_z * target_thickness;
                vertex_visited[vertex_index] = true;
            }
        }

        for (auto cell = triangulation.begin_active(); cell != triangulation.end(); ++cell)
        {
            for (unsigned int vertex_no = 0;
                 vertex_no < GeometryInfo<dim>::vertices_per_cell;
                 ++vertex_no)
            {
                const unsigned int vertex_index = cell->vertex_index(vertex_no);
                if (vertex_visited[vertex_index])
                    cell->vertex(vertex_no)[dim - 1] = conformed_z[vertex_index];
            }
        }
    }

    template <int dim>
    void GridBuilder<dim>::build_box_grid(TriaType& triangulation,
        const user_options& uo, const MPI_Comm& mpi_communicator)
    {
        Point<dim> p1, p2;

        for (unsigned int d = 0; d < dim; ++d)
        {
            p1[d] = uo.box_llp[d];
            p2[d] = uo.box_llp[d] + uo.box_dims[d];
        }

        GridGenerator::subdivided_hyper_rectangle(
            triangulation,
            uo.box_nxyz,
            p1,
            p2,
            true);

        const std::string input_root = join_paths(uo.main_path, uo.input_path);
        const auto top = make_surface_evaluator(uo.top_fnc, mpi_communicator, input_root);
        const auto bottom = make_surface_evaluator(uo.bot_fnc, mpi_communicator, input_root);
        const auto vertical_distribution =
            make_vertical_distribution(uo, mpi_communicator, input_root);
        const unsigned int fallback_level_count =
            static_cast<unsigned int>(uo.box_nxyz[dim - 1]) + 1;
        const unsigned int n_vertical_levels =
            vertical_level_count(uo, vertical_distribution, fallback_level_count);

        conform_to_top_bottom(triangulation,
                              uo,
                              top,
                              bottom,
                              vertical_distribution,
                              p1[dim - 1],
                              p2[dim - 1],
                              n_vertical_levels);

        MPI_Barrier(mpi_communicator);
    }

    template <int dim>
    void GridBuilder<dim>::build_file_grid(TriaType& triangulation,
        const user_options& uo, const MPI_Comm& mpi_communicator)
    {
        BaseTriaType tria2D;
        Triangulation<dim> tria3D;

        read_2d_grid(tria2D, uo);

        const std::string input_root = join_paths(uo.main_path, uo.input_path);
        const auto top = make_surface_evaluator(uo.top_fnc, mpi_communicator, input_root);
        const auto bottom = make_surface_evaluator(uo.bot_fnc, mpi_communicator, input_root);
        const auto vertical_distribution =
            make_vertical_distribution(uo, mpi_communicator, input_root);
        const unsigned int n_vertical_levels =
            vertical_level_count(uo, vertical_distribution, 2);
        AssertThrow(n_vertical_levels >= 2,
                    ExcMessage("The vertical discretization must define at least two levels."));

        GridGenerator::extrude_triangulation(
                tria2D,
                n_vertical_levels,
                100.0,
                tria3D);

        convert_to_parallel(tria3D, triangulation);
        assign_default_boundary_ids(triangulation);
        conform_to_top_bottom(triangulation,
                              uo,
                              top,
                              bottom,
                              vertical_distribution,
                              0.0,
                              100.0,
                              n_vertical_levels);
        MPI_Barrier(mpi_communicator);
    }

    template <int dim>
    void GridBuilder<dim>::read_2d_grid(BaseTriaType &triangulation, const user_options& uo)
    {
        const std::string input_root = join_paths(uo.main_path, uo.input_path);
        const std::string mesh_file = resolve_relative_path(input_root, uo.mesh_file);
        std::ifstream tria_file(mesh_file.c_str());

        if (!tria_file.good())
        {
            throw std::runtime_error("Could not open mesh file: " + mesh_file);
        }

        std::vector<Point<dim - 1>>    vertices;
        std::vector<CellData<dim - 1>> cells;
        SubCellData subcelldata;

        char buffer[512];

        unsigned int n_vertices = 0;
        unsigned int n_cells    = 0;

        tria_file.getline(buffer, 512);
        {
            std::istringstream inp(buffer);
            inp >> n_vertices >> n_cells;
        }

        if (n_vertices == 0 || n_cells == 0)
        {
            throw std::runtime_error("Invalid mesh header in file: " + mesh_file);
        }

        vertices.resize(n_vertices);
        cells.resize(n_cells);

        // Read vertices
        for (unsigned int i = 0; i < n_vertices; ++i)
        {
            if (!tria_file.getline(buffer, 512))
            {
                throw std::runtime_error(
                    "Unexpected end of file while reading vertices from: " +
                    mesh_file);
            }

            std::istringstream inp(buffer);

            Point<basedim> p;
            inp >> p[0] >> p[1];

            if (inp.fail())
            {
                throw std::runtime_error(
                    "Failed reading vertex " + std::to_string(i) +
                    " from mesh file: " + mesh_file);
            }

            vertices[i] = p;
        }

        // Read quadrilateral elements.
        //
        // Input files must list vertices in counter-clockwise polygon order:
        //
        //     3 ---- 2
        //     |      |
        //     0 ---- 1
        //
        // deal.II CellData<2> expects reference-cell order:
        //
        //     2 ---- 3
        //     |      |
        //     0 ---- 1
        for (unsigned int i = 0; i < n_cells; ++i)
        {
            if (!tria_file.getline(buffer, 512))
            {
                throw std::runtime_error(
                    "Unexpected end of file while reading cells from: " +
                    mesh_file);
            }

            std::istringstream inp(buffer);

            unsigned int raw_vertices[GeometryInfo<basedim>::vertices_per_cell];

            for (unsigned int j = 0;
                 j < GeometryInfo<basedim>::vertices_per_cell;
                 ++j)
            {
                inp >> raw_vertices[j];
            }
            if (inp.fail())
            {
                throw std::runtime_error(
                    "Failed reading cell " + std::to_string(i) +
                    " from mesh file: " + mesh_file);
            }

            const double signed_area =
                signed_polygon_area(vertices, raw_vertices);

            if (signed_area <= 0.0)
            {
                throw std::runtime_error(
                    "Cell " + std::to_string(i) +
                    " in mesh file '" + mesh_file +
                    "' is not in counter-clockwise polygon order.");
            }

            cells[i].vertices[0] = raw_vertices[0];
            cells[i].vertices[1] = raw_vertices[1];
            cells[i].vertices[2] = raw_vertices[3];
            cells[i].vertices[3] = raw_vertices[2];
        }

        GridTools::delete_unused_vertices(vertices, cells, subcelldata);

        GridTools::invert_all_negative_measure_cells(vertices, cells);

        GridTools::consistently_order_cells(cells);

        triangulation.create_triangulation(vertices, cells, subcelldata);
    }

    template <int dim>
    double GridBuilder<dim>::signed_polygon_area(
        const std::vector<Point<basedim>> &vertices,
        const unsigned int cell_vertices[4])
    {
        double signed_area2 = 0.0;

        for (unsigned int i = 0; i < 4; ++i)
        {
            const Point<basedim> &a = vertices[cell_vertices[i]];
            const Point<basedim> &b = vertices[cell_vertices[(i + 1) % 4]];

            signed_area2 += a[0] * b[1] - b[0] * a[1];
        }

        return 0.5 * signed_area2;
    }

    template <int dim>
    void GridBuilder<dim>::convert_to_parallel(const Triangulation<dim>& tria3D, TriaType& triangulation)
    {
        const std::vector<Point<dim>> tria3D_vertices = tria3D.get_vertices();
        std::vector<Point<dim>> vertices(tria3D_vertices.size());
        for (unsigned int i = 0; i < tria3D_vertices.size(); ++i)
        {
            for (unsigned int d = 0; d < dim; ++d)
                vertices[i][d] = tria3D_vertices[i][d];
        }

        std::vector<CellData<dim>> cells(tria3D.n_active_cells());

        unsigned int cell_index = 0;
        for (auto cell = tria3D.begin_active();
             cell != tria3D.end(); ++cell)
        {
            for (unsigned int j = 0;
                 j < GeometryInfo<dim>::vertices_per_cell; ++j)
            {
                cells[cell_index].vertices[j] =
                    cell->vertex_index(j);
            }

            ++cell_index;
        }

        SubCellData subcelldata;

        triangulation.create_triangulation(
            vertices,
            cells,
            subcelldata);
    }

    template <int dim>
    void GridBuilder<dim>::assign_default_boundary_ids(TriaType& triangulation)
    {
        for (auto cell = triangulation.begin_active(); cell != triangulation.end(); ++cell)
        {
            if (!cell->is_locally_owned())
                continue;
            for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
            {
                if (cell->face(face)->at_boundary())
                {
                    cell->face(face)->set_all_boundary_ids(static_cast<types::boundary_id>(face));
                }
            }
        }
    }

    template <int dim>
    void GridBuilder<dim>::mark_top_neighborhood_from_cell(const CellIterator& cell, const unsigned int depth)
    {
        if (!cell_has_top_boundary(cell))
            return;

        if (depth == 0)
            return;

        std::set<CellId> visited;
        std::vector<CellIterator> frontier;

        visited.insert(cell->id());
        frontier.push_back(cell);

        for (unsigned int layer = 0; layer < depth; ++layer)
        {
            std::vector<CellIterator> next_frontier;
            for (const auto& current_cell : frontier)
            {
                for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
                {
                    if (current_cell->at_boundary(face))
                        continue;

                    auto neighbor = current_cell->neighbor(face);

                    if (!neighbor->is_active())
                        continue;

                    if (!neighbor->is_locally_owned())
                        continue;

                    const CellId nid = neighbor->id();

                    if (visited.insert(nid).second)
                    {
                        if (!neighbor->refine_flag_set())
                            neighbor->set_refine_flag();

                        next_frontier.push_back(neighbor);
                    }
                }
            }

            frontier.swap(next_frontier);
            if (frontier.empty())
                break;
        }
    }

    template <int dim>
    bool GridBuilder<dim>::cell_has_top_boundary(const CellIterator& cell)
    {
        if (dim == 3)
        {
            return cell->face(5)->at_boundary();
        }
        else
        {
            return false;
        }
    }

    template <int dim>
    bool GridBuilder<dim>::cell_has_other_boundary(const CellIterator& cell)
    {
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
        {
            if (!cell->face(face)->at_boundary())
                continue;

            if (dim == 3)
            {
                if (face == 5)
                    continue;
            }
            return true;
        }
        return false;
    }

    template <int dim>
    bool GridBuilder<dim>::cell_contains_well(const CellIterator& cell, MNWellCollection& mnwells)
    {
        Polygon quad;

        std::vector<double> topZ;
        std::vector<double> bottomZ;
        std::vector<double> Xquad;
        std::vector<double> Yquad;

        std::vector<MNWell*> wells_in_cell;
        std::vector<double> screen_length_inside;
        std::vector<double> well_z_bot;

        quad_and_Zcoords_from_cell<dim>(quad, cell, Xquad, Yquad, topZ, bottomZ);
        //std::cout << "(" << Xquad[0] << "," << Yquad[0] << ") " << std::endl;
        // for (unsigned int i = 0; i < Xquad.size(); ++i)
        //     std::cout << "[(" << Xquad[i] << "," << Yquad[i] << ") " << std::endl;
        // std::cout << "]" << std::endl;

        mnwells.find_wells_in_polygon(
        quad,
        wells_in_cell,
        screen_length_inside,
        well_z_bot,
        Xquad,
        Yquad,
        topZ,
        bottomZ);
        return !wells_in_cell.empty();
    }

    template <int dim>
    bool GridBuilder<dim>::cell_has_top_stream(const CellIterator& cell, StreamCollection<dim>& streams)
    {
        if (streams.empty())
            return false;

        const unsigned int top_face_id = GeometryInfo<dim>::faces_per_cell - 1;

        if (!cell->face(top_face_id)->at_boundary())
            return false;

        Polygon quad;

        std::vector<double> topZ;
        std::vector<double> bottomZ;
        std::vector<double> Xquad;
        std::vector<double> Yquad;

        quad_and_Zcoords_from_cell<dim>(quad, cell, Xquad, Yquad, topZ, bottomZ);

        std::vector<const typename StreamCollection<dim>::Stream *> streams_in_cell;

        streams.find_streams_in_polygon(streams_in_cell, Xquad, Yquad);

        return !streams_in_cell.empty();
    }

    template <int dim>
    void GridBuilder<dim>::refine_triangulation(TriaType &triangulation, const user_options &uo,
        const RefinementTargets &targets, const MPI_Comm &mpi_communicator) {

        if (uo.ref_opt.initial > 0)
        {
            triangulation.refine_global(static_cast<unsigned int>(uo.ref_opt.initial));
            assign_default_boundary_ids(triangulation);
            MPI_Barrier(mpi_communicator);
        }

        const unsigned int n_ref_passes = static_cast<unsigned int>(
            std::max({uo.ref_opt.wells,
                        uo.ref_opt.streams,
                        uo.ref_opt.top,
                        uo.ref_opt.dirichlet,
                        uo.ref_opt.GHB,
                        uo.ref_opt.neumann}));

        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
            std::cout << "Refinement: initial global_active_cells="
                      << triangulation.n_global_active_cells()
                      << std::endl;
        }

        for (unsigned int iref = 0; iref < n_ref_passes; ++iref) {
            const double iref_start_time = MPI_Wtime();

            if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
            {
                std::cout << "---------------------------------------------------------" << std::endl;
                std::cout << "Refinement: starting iref=" << iref
                          << " global_active_cells="
                          << triangulation.n_global_active_cells()
                          << std::endl;
            }

            bool any_marked = false;
            std::set<CellId> well_marked_cell_ids;

            for (auto cell = triangulation.begin_active(); cell != triangulation.end(); ++cell) {
                if (!cell->is_locally_owned())
                    continue;
                if (cell->refine_flag_set())
                    continue;

                if (iref < static_cast<unsigned int>(uo.ref_opt.top) &&
                cell_has_top_boundary(cell))
                {
                    cell->set_refine_flag();
                    mark_top_neighborhood_from_cell(cell, static_cast<unsigned int>(uo.ref_opt.top_depth));
                    any_marked = true;
                    continue;
                }

                if (iref < static_cast<unsigned int>(uo.ref_opt.wells))
                {
                    if (targets.mnwells != nullptr)
                    {
                        if (!targets.mnwells->wells.empty())
                        {
                            if (cell_contains_well(cell, *targets.mnwells))
                            {
                                well_marked_cell_ids.insert(cell->id());
                                cell->set_refine_flag();
                                any_marked = true;
                                continue;
                            }
                        }
                    }
                }

                if (iref < static_cast<unsigned int>(uo.ref_opt.streams) && targets.streams != nullptr &&
                    !targets.streams->empty() && cell_has_top_stream(cell, *targets.streams))
                {
                    cell->set_refine_flag();
                    any_marked = true;
                    continue;
                }

                if (iref < static_cast<unsigned int>(uo.ref_opt.dirichlet) && targets.dirichlet != nullptr &&
                    !targets.dirichlet->get_parts().empty() && cell_has_dirichlet_boundary(cell, *targets.dirichlet))
                {
                    cell->set_refine_flag();
                    any_marked = true;
                    continue;
                }

                if (iref < static_cast<unsigned int>(uo.ref_opt.GHB) && targets.ghb != nullptr &&
                    !targets.ghb->get_parts().empty() && cell_has_ghb_boundary(cell, *targets.ghb))
                {
                    cell->set_refine_flag();
                    any_marked = true;
                    continue;
                }

                // if (iref < static_cast<unsigned int>(uo.ref_opt.neumann) && has_active_predicate(targets.neumann_face) &&
                //     cell_matches_face_predicate(cell, targets.neumann_face))
                // {
                //     cell->set_refine_flag();
                //     any_marked = true;
                //     continue;
                // }
            }

            const bool any_marked_global =
                Utilities::MPI::max(any_marked ? 1 : 0, mpi_communicator) != 0;

            if (!any_marked_global)
            {
                const double iref_elapsed_time =
                    MPI_Wtime() - iref_start_time;

                if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
                {
                    std::cout << "Refinement: finished iref=" << iref
                              << " elapsed_seconds=" << iref_elapsed_time
                              << " global_active_cells="
                              << triangulation.n_global_active_cells()
                              << " marked=0"
                              << std::endl;
                }
                break;
            }

            unsigned int well_marked_flags_missing_before_execute = 0;
            for (auto cell = triangulation.begin_active();
                 cell != triangulation.end();
                 ++cell)
            {
                if (!cell->is_locally_owned())
                    continue;

                if (well_marked_cell_ids.find(cell->id()) ==
                    well_marked_cell_ids.end())
                    continue;

                if (!cell->refine_flag_set())
                    ++well_marked_flags_missing_before_execute;
            }

            triangulation.execute_coarsening_and_refinement();
            assign_default_boundary_ids(triangulation);

            unsigned int still_active_well_marked_cells = 0;
            std::ostringstream still_active_well_marked_details;
            for (auto cell = triangulation.begin_active();
                 cell != triangulation.end();
                 ++cell)
            {
                if (!cell->is_locally_owned())
                    continue;

                if (well_marked_cell_ids.find(cell->id()) !=
                    well_marked_cell_ids.end())
                {
                    ++still_active_well_marked_cells;
                    still_active_well_marked_details
                        << " rank=" << Utilities::MPI::this_mpi_process(mpi_communicator)
                        << " pass=" << iref
                        << " cell_id=" << cell->id().to_string()
                        << " level=" << cell->level()
                        << " center=" << cell->center()
                        << '\n';
                }
            }

            const unsigned int global_still_active_well_marked_cells =
                Utilities::MPI::sum(still_active_well_marked_cells,
                                    mpi_communicator);
            const unsigned int global_missing_before_execute =
                Utilities::MPI::sum(well_marked_flags_missing_before_execute,
                                    mpi_communicator);

            if (global_still_active_well_marked_cells > 0)
            {
                std::cout
                    << "WARNING: " << global_still_active_well_marked_cells
                    << " cells marked by well refinement remained active after "
                    << "execute_coarsening_and_refinement(). Missing flags before execute: "
                    << global_missing_before_execute << std::endl;

                if (still_active_well_marked_cells > 0)
                    std::cout << still_active_well_marked_details.str() << std::flush;
            }

            MPI_Barrier(mpi_communicator);

            const double iref_elapsed_time = MPI_Wtime() - iref_start_time;

            if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
            {
                std::cout << "Refinement: finished iref=" << iref
                          << " elapsed_seconds=" << iref_elapsed_time
                          << " global_active_cells="
                          << triangulation.n_global_active_cells()
                          << std::endl;
            }
        }
    }

    template <int dim>
    bool GridBuilder<dim>::cell_has_dirichlet_boundary(
        const CellIterator &cell,
        const DirichletBoundary<dim> &dirichlet_bc)
    {
        for (unsigned int face = 0;
             face < GeometryInfo<dim>::faces_per_cell;
             ++face)
        {
            if (!cell->face(face)->at_boundary())
                continue;

            if (dirichlet_bc.face_is_dirichlet(cell, face))
                return true;
        }

        return false;
    }

    template <int dim>
    bool GridBuilder<dim>::cell_has_ghb_boundary(
        const CellIterator &cell,
        const GHBBoundary<dim> &ghb_bc)
    {
        for (unsigned int face = 0;
             face < GeometryInfo<dim>::faces_per_cell;
             ++face)
        {
            if (!cell->face(face)->at_boundary())
                continue;

            if (ghb_bc.face_is_ghb(cell, face))
                return true;
        }

        return false;
    }

    template<int dim>
    void GridBuilder<dim>::write_parallel_coarse_tria_to_files(const TriaType &tria, const std::string &prefix, const MPI_Comm &mpi_communicator) {
        int my_rank;
        MPI_Comm_rank(mpi_communicator, &my_rank);

        if (my_rank == 0) {
            const std::string vertex_file = prefix + "_coarse_tria_vertices.dat";
            const std::string cell_file   = prefix + "_coarse_tria_cells.dat";

            const std::vector<Point<dim> > vertices = tria.get_vertices();

            {
                std::ofstream out(vertex_file.c_str());
                AssertThrow(out.good(), dealii::ExcMessage("Cannot write file: " + vertex_file));

                std::cout << "Writing Vertices in " << vertex_file << std::endl;

                out << vertices.size() << " " << dim << "\n";
                out << std::setprecision(17);

                for (unsigned int i = 0; i < vertices.size(); ++i)
                {
                    out << i;
                    for (unsigned int d = 0; d < dim; ++d)
                        out << " " << vertices[i][d];
                    out << "\n";
                }
            }

            {
                std::ofstream out(cell_file.c_str());
                AssertThrow(out.good(), dealii::ExcMessage("Cannot write file: " + cell_file));

                std::cout << "Writing Cells in " << vertex_file << std::endl;

                const unsigned int vpc = GeometryInfo<dim>::vertices_per_cell;

                out << tria.n_global_active_cells() << " " << vpc << "\n";

                unsigned int cid = 0;

                for (typename TriaType::active_cell_iterator
                         cell = tria.begin_active();
                     cell != tria.end();
                     ++cell)
                {
                    out << cid;

                    for (unsigned int j = 0; j < vpc; ++j)
                        out << " " << cell->vertex_index(j);

                    out << "\n";
                    ++cid;
                }
            }
            std::cout << "Wrote coarse triangulation files:\n"
                      << "  " << vertex_file << "\n"
                      << "  " << cell_file << std::endl;
        }
        MPI_Barrier(mpi_communicator);
    }
}

#endif //MESH_GEN_H
