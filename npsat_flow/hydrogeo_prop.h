//
// Created by giorgk on 6/23/26.
//

#ifndef HYDROGEO_PROP_H
#define HYDROGEO_PROP_H

#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>

#include "flow_structures.h"
#include "helper_func.h"
#include "interpolation/interpolation_function.h"

namespace npsat_flow {
    using namespace dealii;

    template <int dim>
    class HydraulicProperties
    {
    public:
        HydraulicProperties() = default;
        void read(const user_options &opt, MPI_Comm comm) {
            const std::string input_root = join_paths(opt.main_path, opt.input_path);
            read_property(opt.hgeo.kx_file, Kx, comm, input_root);
            read_property(opt.hgeo.kz_file, Kz, comm, input_root);
            read_property(opt.hgeo.ss_file, Ss, comm, input_root);
            read_property(opt.hgeo.sy_file, Sy, comm, input_root);
        }

        void set_time_index(const unsigned int itime)
        {
            Kx.set_time_index(itime);
            Kz.set_time_index(itime);
            Ss.set_time_index(itime);
            Sy.set_time_index(itime);
        }

        Tensor<2, dim> conductivity(const Point<dim> &p) const
        {
            Tensor<2, dim> K;
            const double kx = Kx.value(p);
            K[0][0] = kx;

            if (dim >= 2)
                K[1][1] = kx;

            if (dim >= 3)
                K[2][2] = Kz.value(p);

            return K;
        }

        Tensor<2, dim> conductivity_inverse(const Point<dim> &p) const
        {
            const Tensor<2, dim> K = conductivity(p);

            Tensor<2, dim> K_inv;

            for (unsigned int d = 0; d < dim; ++d)
                K_inv[d][d] = 1.0 / K[d][d];

            return K_inv;
        }

        double specific_storage(const Point<dim> &p) const
        {
            return Ss.value(p);
        }

        double specific_yield(const Point<dim> &p) const
        {
            return Sy.value(p);
        }
    private:
        static void read_property(const std::string &file,
                                  InterpolationFunction<dim> &func,
                                  MPI_Comm comm,
                                  const std::string &input_path)
        {
            auto interp = std::make_shared<InterpInterface<dim>>();

            interp->read_master_file(file, 1, comm, input_path);

            func.set_interpolant(interp);
        }

        InterpolationFunction<dim> Kx;
        InterpolationFunction<dim> Kz;
        InterpolationFunction<dim> Ss;
        InterpolationFunction<dim> Sy;
    };


}

#endif //HYDROGEO_PROP_H
