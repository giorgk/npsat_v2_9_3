//
// Created by giorgk on 6/24/26.
//

#ifndef GHB_BC_H
#define GHB_BC_H

#include <deal.II/distributed/tria.h>


#include "boundary_base.h"
#include "../interpolation/interpolation_function.h"

namespace npsat_flow {
    using namespace dealii;

    /**
     * @brief Function pair used by the GHB assembly.
     *
     * head is the external boundary head h_b.
     * conductance is the face conductance density c used in
     * q.n = c * (lambda - h_b).
     */
    template <int dim>
    struct GHBFunctionPair
    {
        const Function<dim> *head = nullptr;
        const Function<dim> *conductance = nullptr;
    };

    /**
     * @brief One user-defined General Head Boundary condition.
     *
     * A GHB part stores one geometry, one custom boundary id, and two
     * interpolation functions:
     * - external head h_b
     * - conductance/leakance density c
     */
    template <int dim>
    class GHBPart
    {
    public:
        BoundaryKind kind;
        BoundaryRegion<dim> region;

        types::boundary_id boundary_id = numbers::invalid_boundary_id;

        std::shared_ptr<InterpInterface<dim>> head_interp;
        std::shared_ptr<InterpInterface<dim>> conductance_interp;

        std::unique_ptr<InterpolationFunction<dim>> head_function;
        std::unique_ptr<InterpolationFunction<dim>> conductance_function;

        GHBPart() = default;

        void set(const BoundaryKind kind_in,
                 BoundaryRegion<dim> region_in,
                 const types::boundary_id boundary_id_in,
                 std::shared_ptr<InterpInterface<dim>> head_interp_in,
                 std::shared_ptr<InterpInterface<dim>> conductance_interp_in)
        {
            kind = kind_in;
            region = std::move(region_in);
            boundary_id = boundary_id_in;
            head_interp = std::move(head_interp_in);
            conductance_interp = std::move(conductance_interp_in);

            head_function = std::make_unique<InterpolationFunction<dim>>(head_interp);
            conductance_function = std::make_unique<InterpolationFunction<dim>>(conductance_interp);
        }

        bool is_initialized() const
        {
            return boundary_id != numbers::invalid_boundary_id &&
                   head_interp != nullptr &&
                   conductance_interp != nullptr &&
                   head_function != nullptr &&
                   conductance_function != nullptr;
        }

        void set_time_index(const unsigned int time_index)
        {
            if (head_function)
                head_function->set_time_index(time_index);

            if (conductance_function)
                conductance_function->set_time_index(0);
        }
    };

    /**
     * @brief Manager for all General Head Boundary conditions.
     *
     * File format:
     *
     * @code
     * N
     * TYPE nv head_data_file conductance_data_file
     * x1 y1
     * ...
     * xnv ynv
     * @endcode
     *
     * TYPE must be TOP, BOT, EDGE, or EDGETOP.
     *
     * The returned function map is intended for assemble_system(), where the
     * GHB contributes a Robin term to the condensed trace system.
     */
    template <int dim>
    class GHBBoundary
    {
    public:
        using Part = GHBPart<dim>;
        using FunctionPair = GHBFunctionPair<dim>;
        using TriaType = parallel::distributed::Triangulation<dim>;
        using CellIterator = typename TriaType::active_cell_iterator;

        GHBBoundary() = default;

        void clear()
        {
            parts.clear();
            function_map.clear();
        }

        void set_lateral_matching_tolerances(const double half_width,
                                             const double min_overlap_length)
        {
            lateral_half_width = half_width;
            min_lateral_overlap = min_overlap_length;
        }

        void read_data(const std::string &filename,
                       const double head_factor = 1.0,
                       const double conductance_factor = 1.0,
                       MPI_Comm comm = MPI_COMM_WORLD,
                       const types::boundary_id first_custom_id = 1000,
                       const std::string &input_path = "")
        {
            clear();

            const std::string boundary_file = resolve_relative_path(input_path, filename);
            std::string boundary_text;
            read_text_file_mpi(boundary_file, comm, boundary_text);
            std::istringstream in(boundary_text);

            unsigned int n_parts = 0;
            in >> n_parts;
            if (!in)
                throw std::runtime_error("Invalid GHB file: " + boundary_file);
            parts.resize(n_parts);

            for (unsigned int i = 0; i < n_parts; ++i)
            {
                std::string type_string;
                int nv = 0;
                std::string head_data_file;
                std::string conductance_data_file;

                in >> type_string >> nv >> head_data_file >> conductance_data_file;

                const BoundaryKind kind = boundary_kind_from_string(type_string);
                BoundaryRegion<dim> region = read_boundary_region(in, kind, nv);

                auto head_interp = std::make_shared<InterpInterface<dim>>();
                head_interp->read_master_file(head_data_file,
                                              head_factor,
                                              comm,
                                              input_path,
                                              lateral_half_width,
                                              min_lateral_overlap);

                auto conductance_interp = std::make_shared<InterpInterface<dim>>();
                conductance_interp->read_master_file(conductance_data_file,
                                                     conductance_factor,
                                                     comm,
                                                     input_path,
                                                     lateral_half_width,
                                                     min_lateral_overlap);

                const auto bid = static_cast<types::boundary_id>(first_custom_id + i);

                parts[i].set(kind,
                             std::move(region),
                             bid,
                             std::move(head_interp),
                             std::move(conductance_interp));
            }

            build_function_map();
        }

        void build_function_map()
        {
            function_map.clear();

            for (const auto &part : parts)
            {
                if (!part.is_initialized())
                    continue;

                function_map[part.boundary_id] =
                    FunctionPair{part.head_function.get(),
                                 part.conductance_function.get()};
            }
        }

        const std::map<types::boundary_id, FunctionPair> &
        get_function_map() const
        {
            return function_map;
        }

        std::map<types::boundary_id, FunctionPair> &
        get_function_map()
        {
            return function_map;
        }

        void set_time_index(const unsigned int time_index)
        {
            for (auto &part : parts)
                part.set_time_index(time_index);
        }

        /**
         * @brief Assign custom GHB ids to matching triangulation boundary faces.
         *
         * If overwrite_existing_custom_ids is false, faces with user_index() != 0
         * are left unchanged. This lets Dirichlet boundaries take precedence when
         * both managers are used.
         */
        void assign_to_triangulation(parallel::distributed::Triangulation<dim> &triangulation,
                                     const bool overwrite_existing_custom_ids = false)
        {
            for (auto cell = triangulation.begin_active();
                 cell != triangulation.end();
                 ++cell)
            {
                if (!(cell->is_locally_owned() || cell->is_ghost()))
                    continue;

                for (unsigned int iface = 0;
                     iface < GeometryInfo<dim>::faces_per_cell;
                     ++iface)
                {
                    auto face = cell->face(iface);

                    if (!face->at_boundary())
                        continue;

                    if (!overwrite_existing_custom_ids && face->user_index() != 0)
                        continue;

                    if (overwrite_existing_custom_ids)
                    {
                        face->set_all_boundary_ids(iface);
                        face->set_user_index(0);
                    }

                    const int ipart = get_ghb_part_index_for_tria_face(cell, iface);
                    if (ipart < 0)
                        continue;

                    const Part &part = parts[static_cast<unsigned int>(ipart)];

                    face->set_all_boundary_ids(part.boundary_id);
                    face->set_user_index(2);
                }
            }
        }

        int get_ghb_id_for_face(const CellIterator &cell,
                                const unsigned int iface) const
        {
            const int ipart = get_ghb_part_index_for_tria_face(cell, iface);
            if (ipart < 0)
                return -1;

            return static_cast<int>(parts[static_cast<unsigned int>(ipart)].boundary_id);
        }

        bool face_is_ghb(const CellIterator &cell,
                         const unsigned int iface) const
        {
            return get_ghb_part_index_for_tria_face(cell, iface) >= 0;
        }

        const std::vector<Part> &get_parts() const
        {
            return parts;
        }

        std::vector<Part> &get_parts()
        {
            return parts;
        }

    private:
        std::vector<Part> parts;
        std::map<types::boundary_id, FunctionPair> function_map;

        double lateral_half_width = 20.0;
        double min_lateral_overlap = 5.0;

        BoundaryRegion<dim> read_boundary_region(std::istream &in,
                                                 const BoundaryKind,
                                                 const int nv) const
        {
            if (nv <= 0)
                throw std::runtime_error("GHB boundary entries must have nv > 0.");

            std::vector<Point<dim>> pts(static_cast<unsigned int>(nv));

            for (int i = 0; i < nv; ++i)
            {
                double x, y;
                in >> x >> y;

                Point<dim> p;
                p[0] = x;
                p[1] = y;

                pts[static_cast<unsigned int>(i)] = p;
            }

            BoundaryRegion<dim> region;
            region.points = std::move(pts);

            if (region.points.size() == 2)
                region.set_rectangle(region.points);
            else
                region.set_polygon(region.points);

            return region;
        }

        template <typename CellIterator>
        int get_ghb_part_index_for_face(const CellIterator &cell,
                                        const unsigned int iface) const
        {
            for (unsigned int ipart = 0; ipart < parts.size(); ++ipart)
            {
                const Part &part = parts[ipart];

                if (!part.is_initialized())
                    continue;

                if (!face_matches_kind(cell, iface, part.kind))
                    continue;

                if (!face_matches_region(cell, iface, part.region, part.kind))
                    continue;

                if (!face_matches_lateral_interpolant(cell, iface, part))
                    continue;

                return static_cast<int>(ipart);
            }

            return -1;
        }

        int get_ghb_part_index_for_tria_face(const CellIterator &cell,
                                             const unsigned int iface) const
        {
            for (unsigned int ipart = 0; ipart < parts.size(); ++ipart)
            {
                const Part &part = parts[ipart];

                if (!part.is_initialized())
                    continue;

                if (!face_matches_kind(cell, iface, part.kind))
                    continue;

                if (!face_matches_region(cell, iface, part.region, part.kind))
                    continue;

                if (!face_matches_lateral_interpolant(cell, iface, part))
                    continue;

                return static_cast<int>(ipart);
            }

            return -1;
        }

        template <typename CellIterator>
        bool face_matches_kind(const CellIterator &cell,
                               const unsigned int iface,
                               const BoundaryKind kind) const
        {
            const unsigned int top_face =
                GeometryInfo<dim>::faces_per_cell - 1;

            const unsigned int bot_face =
                GeometryInfo<dim>::faces_per_cell - 2;

            if (kind == BoundaryKind::TOP)
                return iface == top_face ||
                       cell->face(iface)->boundary_id() == top_face;

            if (kind == BoundaryKind::BOT)
                return iface == bot_face ||
                       cell->face(iface)->boundary_id() == bot_face;

            if (kind == BoundaryKind::EDGE ||
                kind == BoundaryKind::EDGETOP)
            {
                const bool vertical_face =
                    iface != top_face && iface != bot_face;

                if (!vertical_face)
                    return false;

                if (kind == BoundaryKind::EDGETOP &&
                    !cell->face(top_face)->at_boundary())
                    return false;

                return true;
            }

            return false;
        }

        template <typename CellIterator>
        bool face_matches_region(const CellIterator &cell,
                                 const unsigned int iface,
                                 const BoundaryRegion<dim> &region,
                                 const BoundaryKind kind) const
        {
            const std::vector<Point<dim>> face_points = get_face_points(cell,
                                                                        iface);

            if (kind == BoundaryKind::TOP || kind == BoundaryKind::BOT)
            {
                const boost_polygon face_polygon = make_face_polygon(face_points);

                return bg::intersects(face_polygon, region.polygon) ||
                       bg::within(face_polygon, region.polygon) ||
                       bg::within(region.polygon, face_polygon);
            }

            if (kind == BoundaryKind::EDGE || kind == BoundaryKind::EDGETOP)
            {
                for (const auto &p : face_points)
                    if (region.contains_point(p))
                        return true;

                return false;
            }

            return false;
        }

        template <typename CellIterator>
        static std::vector<Point<dim>> get_face_points(const CellIterator &cell,
                                                       const unsigned int iface)
        {
            std::vector<Point<dim>> face_points;

            for (unsigned int iv = 0;
                 iv < GeometryInfo<dim>::vertices_per_face;
                 ++iv)
            {
                face_points.push_back(cell->face(iface)->vertex(iv));
            }

            return face_points;
        }

        static boost_polygon make_face_polygon(const std::vector<Point<dim>> &face_points)
        {
            boost_polygon poly;

            for (const auto &p : face_points)
                bg::append(poly.outer(), boost_point(p[0], p[1]));

            bg::append(poly.outer(),
                       boost_point(face_points.front()[0],
                                   face_points.front()[1]));

            bg::correct(poly);

            return poly;
        }

        static std::pair<Point<dim>, Point<dim>>
        representative_face_edge_xy(const std::vector<Point<dim>> &face_points)
        {
            constexpr double xy_tolerance = 1.0e-10;

            std::vector<Point<dim>> unique_xy_points;
            unique_xy_points.reserve(2);

            for (const auto &candidate : face_points)
            {
                bool already_seen = false;

                for (const auto &existing : unique_xy_points)
                {
                    const double dx = candidate[0] - existing[0];
                    const double dy = candidate[1] - existing[1];

                    if (std::sqrt(dx * dx + dy * dy) <= xy_tolerance)
                    {
                        already_seen = true;
                        break;
                    }
                }

                if (!already_seen)
                    unique_xy_points.push_back(candidate);
            }

            AssertThrow(unique_xy_points.size() == 2,
                        ExcMessage("Expected an extruded lateral face to have exactly two unique XY vertices."));

            const double dx = unique_xy_points[1][0] - unique_xy_points[0][0];
            const double dy = unique_xy_points[1][1] - unique_xy_points[0][1];
            const double length = std::sqrt(dx * dx + dy * dy);

            AssertThrow(length > xy_tolerance,
                        ExcMessage("Extruded lateral face has degenerate projected XY edge."));

            return {unique_xy_points[0], unique_xy_points[1]};
        }

        template <typename CellIterator>
        bool face_matches_lateral_interpolant(const CellIterator &cell,
                                              const unsigned int iface,
                                              const Part &part) const
        {
            if (part.kind != BoundaryKind::EDGE &&
                part.kind != BoundaryKind::EDGETOP)
                return true;

            const auto edge =
                representative_face_edge_xy(get_face_points(cell, iface));

            const bool head_requires_overlap =
                part.head_interp->requires_lateral_face_overlap();
            const bool conductance_requires_overlap =
                part.conductance_interp->requires_lateral_face_overlap();

            if (!head_requires_overlap && !conductance_requires_overlap)
                return true;

            const bool head_matches =
                !head_requires_overlap ||
                part.head_interp->face_matches_lateral_overlap(edge.first,
                                                               edge.second);
            const bool conductance_matches =
                !conductance_requires_overlap ||
                part.conductance_interp->face_matches_lateral_overlap(edge.first,
                                                                      edge.second);

            return head_matches && conductance_matches;
        }
    };
}

#endif //GHB_BC_H
