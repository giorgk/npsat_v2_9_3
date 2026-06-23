//
// Created by giorgk on 6/23/26.
//

#ifndef INTERP_INTERFACE_H
#define INTERP_INTERFACE_H

#include <deal.II/base/point.h>

#include "../helper_func.h"
#include "interp_helpers.h"

namespace npsat_flow{
    using namespace dealii;

    template <int dim>
    class InterpInterface{
    public:
        InterpInterface() = default;

        double interpolate(const Point<dim> &p, unsigned int time_index) const
        {
            for (const auto &interp : interpolants)
            {
                if (interp->contains(p))
                    return interp->interpolate(p, time_index);
            }

            return default_value;
        }

        /**
         * @brief Read an interpolation master file and construct its regions.
         *
         * The master file lists interpolation regions. For each region, this
         * function reads the region geometry, creates the requested spatial
         * interpolant, reads its spatial definition file, and then reads the
         * corresponding value time-series file.
         *
         * Region ownership is checked in the XY plane through InterpRegion2D,
         * even when dim == 3. The first region containing a query point is used.
         *
         * Master file format:
         *
         * @code
         * n_regions
         *
         * n_vertices interpolation_type spatial_file values_file
         * region_x0 region_y0
         * region_x1 region_y1
         * ...
         * @endcode
         *
         * For each region:
         *
         * - n_vertices == 2: rectangular region. Exactly two XY points are read,
         *   interpreted as opposite rectangle corners.
         * - n_vertices > 2: polygon region. Exactly n_vertices XY vertices are read.
         * - n_vertices < 2: invalid region definition.
         *
         * Supported interpolation_type values are:
         *
         * - CONST
         * - SCATTER
         * - GRIDDED
         * - LATERAL_POLYLINE
         * - LATERAL
         *
         * spatial_file is passed to the interpolant's read_spatial_data().
         * values_file is passed to the interpolant's read_values().
         * Relative file names are resolved against input_path.
         *
         * Example:
         *
         * @code
         * 2
         *
         * 2 CONST recharge_region.dat recharge_values.dat
         * 0 0
         * 1000 1000
         *
         * 4 GRIDDED grid_definition.dat grid_values.dat
         * 0 0
         * 1000 0
         * 1000 1000
         * 0 1000
         * @endcode
         *
         * @param filename Master-file path, or path relative to input_path.
         * @param factor Multiplicative factor applied to each values file.
         * @param comm MPI communicator used by the interpolants while reading.
         * @param input_path Base path used to resolve relative file names.
         */
        void read_master_file(const std::string &filename,
                              double factor,
                              MPI_Comm comm,
                              const std::string &input_path = "",
                              double lateral_polyline_half_width = 20.0,
                              double lateral_polyline_min_overlap = 5.0);
        std::int64_t n_times() const;
        bool requires_lateral_face_overlap() const;
        bool face_matches_lateral_overlap(const Point<dim> &a,
                                          const Point<dim> &b) const;

    private:
        static InterpRegion2D<dim> read_region(std::istream &in, const int n_vertices);
        static double region_area(const InterpRegion2D<dim> &region);
        std::unique_ptr<SpatialInterpolantBase<dim>> create_interpolant(const std::string &interp_type,
                                                                        InterpRegion2D<dim> region,
                                                                        double lateral_polyline_half_width,
                                                                        double lateral_polyline_min_overlap);
        std::vector<std::unique_ptr<SpatialInterpolantBase<dim>>> interpolants;

        double default_value = 0.0;
    };

    template <int dim>
    void InterpInterface<dim>::read_master_file(const std::string& filename,
                                                double factor,
                                                MPI_Comm comm,
                                                const std::string &input_path,
                                                const double lateral_polyline_half_width,
                                                const double lateral_polyline_min_overlap){

        const std::string master_file = resolve_relative_path(input_path, filename);

        //std::string master_text;
        //read_text_file_mpi(master_file, comm, master_text);
        //std::istringstream in(master_text);
    }

}

#endif //INTERP_INTERFACE_H
