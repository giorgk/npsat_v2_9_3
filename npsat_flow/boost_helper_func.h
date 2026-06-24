//
// Created by giorgk on 6/24/26.
//

#ifndef BOOST_HELPER_FUNC_H
#define BOOST_HELPER_FUNC_H

#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

#include "flow_structures.h"

namespace npsat_flow{
    using namespace dealii;
    namespace bg = boost::geometry;
    //namespace bgi = boost::geometry::index;
    using Point2D = bg::model::point<double, 2, bg::cs::cartesian>;
    using WellValue = std::pair<Point2D, std::size_t>;
    using Polygon = bg::model::polygon<Point2D>;

    typedef bg::model::d2::point_xy<double> BoostPoint2D;
    typedef bg::model::polygon<BoostPoint2D> BoostPolygon2D;

    template <int dim, typename CellIterator>
    void quad_and_Zcoords_from_cell(
        Polygon& quad,
        const CellIterator &cell,
        std::vector<double> &X, std::vector<double> &Y,
        std::vector<double> &top, std::vector<double> &bottom)
    {
        X.clear();
        Y.clear();
        top.clear();
        bottom.clear();
        quad.outer().clear();

        const auto bot_face = cell->face(4);
        const auto top_face = cell->face(5);

        static constexpr double eps = 1e-4;
        // Extract the 4 vertices of the face
        for (unsigned int v : face_order)
        {
            const Point<dim> &p_top = top_face->vertex(v);
            const Point<dim> &p_bot = bot_face->vertex(v);
            // Check XY consistency between top and bottom faces
            if (std::abs(p_top[0] - p_bot[0]) > eps || std::abs(p_top[1] - p_bot[1]) > eps)
            {
                AssertThrow(false, ExcMessage("Top and bottom face XY mismatch exceeds tolerance"));
            }

            X.push_back(p_top[0]);
            Y.push_back(p_top[1]);
            top.push_back(p_top[2]);
            bottom.push_back(p_bot[2]);

            // Project to XY plane
            quad.outer().push_back(Point2D(p_bot[0], p_bot[1]));
        }

        // Close the polygon (Boost requires last == first)
        const Point<3> &p0 = bot_face->vertex(0);
        quad.outer().push_back(Point2D(p0[0], p0[1]));
        // Optional: normalize orientation (Boost likes CCW)
        bg::correct(quad);
    }

    template <int dim>
    void create_boost_polygon_from_face(
        const typename DoFHandler<dim>::face_iterator& face,
        BoostPolygon2D& polygon)
    {
        // Clear the polygon
        bg::clear(polygon);

        // Get number of vertices on face (should be 4 for hexahedra)
        const unsigned int n_vertices = face->n_vertices();

        // Create outer ring of polygon
        BoostPolygon2D::ring_type ring;

        // Add vertices in order (assuming standard counter-clockwise order)
        // For quadrilateral faces, vertices are typically ordered:
        // 0: bottom-left, 1: bottom-right, 2: top-right, 3: top-left

        for (unsigned int v = 0; v < n_vertices; ++v)
        {
            const Point<dim>& vertex = face->vertex(v);
            ring.push_back(BoostPoint2D(vertex[0], vertex[1]));
        }

        // Close the ring (add first point again)
        if (n_vertices > 0)
        {
            const Point<dim>& first_vertex = face->vertex(0);
            ring.push_back(BoostPoint2D(first_vertex[0], first_vertex[1]));
        }
        // Set the outer ring
        bg::append(polygon, ring);

        // Correct orientation if needed (Boost expects outer ring counter-clockwise)
        bg::correct(polygon);
    }


}

#endif //BOOST_HELPER_FUNC_H
