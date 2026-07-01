//
// Created by giorgk on 6/27/26.
//

#ifndef CACHED_VELOCITY_H
#define CACHED_VELOCITY_H
namespace npsat_trace {
    using namespace dealii;

    template<int dim>
    class CellVelocityCacheRT0Split3D {
    public:
        static_assert(dim == 3, "This cache is designed for 3D.");

        using CellIt = typename DoFHandler<dim>::active_cell_iterator;


    };

}

#endif //CACHED_VELOCITY_H
